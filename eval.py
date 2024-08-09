import os
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

prompt_for_multi_choice = """\
你现在是一位传统中医领域的专家，拥有多年的教学和临床经验，也担任过多年的传统中医入学考试出题人。接下来我会给你一道传统中医领域知识题，请根据题干给出正确答案。

可供选择的答案选项总共有5项，分别是A、B、C、D、E，但是正确答案只有1项，如A或B。

你首先一步一步的推理，然后根据题干给出正确答案，不需要给出推理过程，除了答案不要给出其他任何信息。

接下来是2道传统中医领域知识题例子：
题干：
患者，男，50岁。眩晕欲仆，头摇而痛，项强肢颤，腰膝疫软，舌红苔薄白，脉弦有力。其病机是
A．肝阳上亢
B．肝肾阴虚
C．肝阳化风
D．阴虚风动
E．肝血不足
答案：C

题干：
按照《处方管理办法（试行）》文件，处方是医师为患者开具的一种
A.医疗诊断证明
B.医疗用药的医疗文书
C.用药的标准规范
D.用药的技术规范
E.资质证明文件
答案：B

接下来是你要回答的传统中医领域知识题：
题干：
{instruction}
"""

prompt_for_multi_answers = """\
你现在是一位传统中医领域的专家，拥有多年的教学和临床经验，也担任过多年的传统中医入学考试出题人。接下来我会给你一道传统中医领域知识题，请根据题干给出正确答案。

可供选择的答案选项总共有5项，分别是A、B、C、D、E，但是正确答案不一定只有1项，可能是多项，如AB或ACD或ABCDE。

你首先一步一步的推理，然后根据题干给出正确答案，不需要给出推理过程，除了答案不要给出其他任何信息。

接下来是2个例子：
题干：
为保证应急用血，医疗机构可以临时采集血液，但必须同时符合以下哪些条件,并应当在临时采集血液后10日内将情况报告县级以上人民政府卫生行政部门 
A.危及患者生命，急需输血
B.所在地血站无法及时提供血液，且无法及时从其他医疗机构调剂血液，而其他医疗措施不能替代输血治疗
C.具备开展交叉配血及乙型肝炎病毒表面抗原、丙型肝炎病毒抗体、艾滋病病毒抗体和梅毒螺旋体抗体的检测能力
D.遵守采供血相关操作规程和技术标准。
答案：ABCD

题干：
下列哪些选项属于医院焦虑反应
A.对以往感兴趣的事物失去兴趣
B.担心治疗出现错
C.有时出现莫名的恐慌感
D.对自己的未来感到一片黑暗
E.手术前患者坐立不安
答案：BCE

接下来是你要回答的传统中医领域知识题：
题干：
{instruction}
"""

prompt_for_true_or_false = """\
你现在是一位传统中医领域的专家，拥有多年的教学和临床经验，也担任过多年的传统中医入学考试出题人。接下来我会给你一道传统中医领域知识题，请根据题干给出正确答案。

可供选择的答案选项总共有2项，分别是正确、错误，但是正确答案只有1项，如正确或错误，也就是判断题。

你首先一步一步的推理，然后根据题干给出正确答案，不需要给出推理过程，除了答案不要给出其他任何信息。

接下来是2个例子：
题干：中年以上，阴虚阳亢，风阳上扰的眩晕，往往有中风的可能。
答案：正确

题干：胸闷心烦不寐，泛恶；暖气，伴头重目眩，口苦，舌红苔黄腻，脉滑数，辨证为肝郁化火。
答案：错误

接下来是你要回答的传统中医领域知识题：
题干：{instruction}
"""

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
    base_url="http://127.0.0.1:11435/v1",
)


def answer_from_llm(instruction: str, model: str) -> str:
    chat_completion = client.chat.completions.create(
        timeout=60,
        messages=[{"role": "user", "content": f"{instruction}",}],
        model=model,
        temperature=0.0,
        max_tokens=256,
    )
    return chat_completion.choices[0].message.content


def process_item(index, item, model):
    category = None
    gt_answer = item["output"]
    if item["output"] == "正确" or item["output"] == "错误":  # 判断题
        category = "判断题"
        full_predicted_answer = answer_from_llm(
            prompt_for_true_or_false.format(instruction=item["instruction"]),
            model,
        )
        if '错误' in full_predicted_answer:
            predicted_answer = '错误'
        elif '正确' in full_predicted_answer:
            predicted_answer = '正确'
        else:
            predicted_answer = ''
    elif len(item["output"]) > 1:  # 多选题
        category = "多选题"
        full_predicted_answer = answer_from_llm(
            prompt_for_multi_answers.format(instruction=item["instruction"]),
            model,
        )
        try:
            predicted_answer = re.search(r'[A-Z]{1,5}', full_predicted_answer).group()
        except:
            predicted_answer = ''
    else:  # 单选题
        category = "单选题"
        full_predicted_answer = answer_from_llm(
            prompt_for_multi_choice.format(instruction=item["instruction"]),
            model,
        )
        try:
            predicted_answer = re.search(r'[A-Z]+', full_predicted_answer).group()
        except:
            predicted_answer = ''
    return index, category, predicted_answer, gt_answer


def main():
    question_amount = {"单选题": 32018, "多选题": 131, "判断题": 96} # 先验知识
    ds = load_dataset("Monor/hwtcm", split="train")
    result = {}
    
    for model in [
        "qwen:7b-chat",
        "qwen:14b-chat",
        "qwen2:7b-instruct",
        "llama3:8b",
        "phi3:14b-instruct",
        "aya:8b",
        "mistral:7b-instruct",
    ]:
        benchmarking_log_path = f"benchmarking_of_{model.replace(':', '_')}.txt"
        if os.path.exists(benchmarking_log_path): os.remove(benchmarking_log_path)
        
        result[model] = {}
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_item, index, item, model) for index, item in enumerate(ds)]
            for future in tqdm(as_completed(futures), total=len(futures), desc=model):
                index, category, predicted_answer, gt_answer = (future.result())

                if category not in result[model]:
                    result[model][category] = {'总量': question_amount[category], '正确量': 0, '正确率': 0}

                if predicted_answer == gt_answer:
                    result[model][category]['正确量'] += 1
                    result[model][category]['正确率'] = result[model][category]['正确量'] / result[model][category]['总量']

                with open(benchmarking_log_path, "a+", encoding="utf-8") as w:
                    w.write(f"{index} {1 if predicted_answer == gt_answer else 0}\n")


    with open('summary.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
