# Licensed under the MIT license.

from eval_src.toolkit_for_MATH.latex_answer_check import latex_answer_check as latex_equiv
import math
import os, json, re
import torch
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from fuzzywuzzy import fuzz, process
from eval_src.toolkit_for_MATH.parsing_lib import *

class Evaluator:
    def __init__(self) -> None:
        self.answer_marker = "answer is"

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata

            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None

    def validate_completion(self, completion: str) -> bool:
        if self.answer_marker.lower() in completion.lower():
            return True

        return False

    def isolate_answer(self, text: str):
        if text is None:
            return None

        assert isinstance(text, str)
        text = text.lower()
        split_ans = text.split(self.answer_marker.lower())
        if len(split_ans) > 1:
            ans = split_ans[-1].replace(":", "").strip()
            # extract_ans_temp = ans.split(".\n")[0].strip()
            extract_ans_temp = ans.split(".\n")[0].split(". ")[0].split("(")[0].strip()    # TODO: 我们添加的
            if extract_ans_temp == "":
                extract_ans_temp = split_ans[-1].replace(":", "").replace(".", "").strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip().strip("\n")
            if extract_ans == "\\" or "$" or "$$":
                extract_ans = ans.split(".\n")[0].rstrip(".").strip()
                # print(f"ans: {ans}")
                # print(f"extract_ans: {extract_ans}")
            return extract_ans
        else:
            return text

    def find_most_confident_answer(self, user_question: str, completions: List[str], prompt_template: str, examples: str, io, entail_model=None, entail_tokenizer=None, prior_weights: List[float] = None):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)  # list 作为参数传递给 defaultdict，表示如果访问的键在字典中不存在，将自动为该键生成一个默认的空列表 ([])
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_answer_from_model_completion(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass
        
        if None in answer2ids:
            del answer2ids[None]
        if None in answer2completions:
            del answer2completions[None]
        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])

            return (
                self.extract_answer_from_model_completion(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            sorted_answers = sorted(answer2completions.keys(), key=lambda x: len(answer2completions[x]), reverse=True)
            if len(sorted_answers) > 1 and len(answer2completions[sorted_answers[0]])  == len(answer2completions[sorted_answers[1]]):
                if entail_model is None:
                    check_io_input = prompt_template.format(
                        examples=examples,
                        Question=user_question,
                        Output1=answer2completions[sorted_answers[0]][0],
                        Output2=answer2completions[sorted_answers[1]][0]
                    )
                    check_output_list = io.generate(
                        model_input=check_io_input, max_tokens=10, num_return=5, stop_tokens=["\n", "\n\n"]
                    )
                    try:
                        check_output_list = [z.strip()[0] for z in check_output_list]  # TODO: 不是不会判断，是只会输出1 (prompt重新设计)
                        one_count = check_output_list.count('1')
                    except:
                        one_count = 5
                    if one_count >= 3:
                        most_confident_answer = sorted_answers[0]
                    else:
                        most_confident_answer = sorted_answers[1]
                else:
                    max_prob = 0
                    best_answer = None
                    io_output_list = [answer2completions[sorted_answers[0]][0], answer2completions[sorted_answers[1]][0]]
                    for answer in io_output_list:
                        inputs = entail_tokenizer(user_question, answer, return_tensors="pt", truncation=True)
                        outputs = entail_model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1).squeeze()
                        
                        entailment_prob = probs[2].item()  # obtain the probability of the "Entailment" class
                        # print(f"answer: {answer}\nprob: {entailment_prob}")
                        if entailment_prob > max_prob:
                            max_prob = entailment_prob
                            best_answer = answer

                    print("The best answer:", best_answer)
                    print("Support Probability:", max_prob)
                    ind = io_output_list.index(best_answer)
                    most_confident_answer = sorted_answers[0] if ind == 0 else sorted_answers[1]
            
            assert (
                len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            
            # confidence定义为多数答案的数量占总数的比例，这可能就是文中提到的一句话
            # "To compute the Q(sd, ad) for the terminal node, we use the likelihood (confidence) of self-consistency majority voting as the reward value"
            confidence = len(answer2completions[most_confident_answer]) / len(completions)
            assert confidence > 0
            return (
                most_confident_answer,
                answer2completions[most_confident_answer][0],  # TODO: 从多数答案对应的CoT中选择第一个（中间过程有错误就没办法检测出来）
                answer2ids[most_confident_answer][0],
                confidence,
            )

    def stochastic_select_answer(self, completion2score, answer2completions, completions):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1

        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]

        top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:1]
        answers, scores = zip(*top_answers)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
        except:
            selected_answer = random.choices(answers, k=1)[0]

        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def stochastic_calculate_completion_scores(self, prior_weights, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count

        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = prior_weights[idx] if prior_weights is not None else 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score

    def stochastic_select_response(self, completion2score, completions):
        sorted_completions = sorted(completion2score.items(), key=lambda x: x[1], reverse=True)[:1]
        completions, scores = zip(*sorted_completions)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            sampled_completion = random.choices(completions, weights=probabilities, k=1)[0]
        except:
            sampled_completion = random.choices(completions, k=1)[0]
        confidence = completion2score[sampled_completion]
        most_confident_answer = self.extract_answer_from_model_completion(sampled_completion)
        id_of_most_confident = completions.index(sampled_completion)
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def stochastic_find_most_confident_answer(
        self,
        completions: List[str],
        prior_weights: List[float] = None,
    ):

        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None

        completion2score = self.stochastic_calculate_completion_scores(prior_weights, answer2completions)

        most_confident_answer, sampled_completion, id_of_most_confident, confidence = self.stochastic_select_response(
            completion2score, completions
        )
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError


class GSM8KEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False

        try:
            if abs(float(search_for_numbers(answer_a)[0]) - float(search_for_numbers(answer_b)[0])) < 1e-6:
                return True
        except:
            a = 1
        
        return correct

    def extract_answer_from_gold_solution(self, solution: str | float):
        """Extract the answer from the gold solution."""
        if isinstance(solution, float):
            return str(solution)
        return solution.split("#### ")[-1].strip()

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None

        assert isinstance(completion, str)

        preds = completion
        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return None
        else:
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]

        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]

        pred = pred.replace(",", "").replace("\n", "")
        is_number, pred = self._is_number(pred)
        if is_number:
            return pred
        else:
            return None


GSM8KHARDEvaluator = GSM8KEvaluator
MULTIARITHEvaluator = GSM8KEvaluator

class MATHEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
        
    def find_box(self, pred_str: str):
        ans = pred_str.split('boxed')[-1]
        if not ans:
            return ""    # origin:     return ""
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        return a

    # 替换指定字符的函数
    def clean_answer(self, answer, markers):
        for marker in markers:
            answer = answer.replace(marker, "")
        # 移除匹配的括号
        for open_bracket, close_bracket in [("(", ")"), ("[", "]"), ("{", "}")]:
            if open_bracket in answer and close_bracket in answer:
                answer = answer.replace(open_bracket, "").replace(close_bracket, "")
        return answer
    
    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        if answer_a == "" or answer_b == "":
            return False

        answer_a = answer_a.strip()
        answer_b = answer_b.strip()

        if answer_a.lower() == answer_b.lower():
            return True
        
        try:
            res = latex_equiv(answer_a, answer_b)
        except Exception as e:
            print(e)
            res = False
        
        # TODO: new_added
        if not res and answer_a != "\\":
            try:
                answer_a = answer_a.replace("$", "")
                answer_b = answer_b.replace("$", "")
                if "sqrt" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if a_list == b_list:
                        return True
                elif "frac" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    a_list_strip, b_list_strip = [], []
                    for x in a_list:
                        a_list_strip += x.split("/")
                    for x in b_list:
                        b_list_strip += x.split("/")
                    if a_list_strip == b_list_strip or "".join(a_list_strip) == "".join(b_list_strip):
                        return True
                    elif len(a_list_strip) == 2 and len(b_list_strip) == 1:
                        if abs(float(a_list_strip[0])/float(a_list_strip[1]) - float(b_list_strip[0])) < 1e-6:
                            return True
                    elif len(a_list_strip) == 1 and len(b_list_strip) == 2:
                        if abs(float(b_list_strip[0])/float(b_list_strip[1]) - float(a_list_strip[0])) < 1e-6:
                            return True
                    elif len(a_list_strip) == 1 and len(b_list_strip) == 1:
                        try:
                            b_list_strip = float(b_list_strip[0][0]) / float(b_list_strip[0][1:])
                            if abs(float(a_list_strip[0]) - b_list_strip) < 1e-3:
                                return True
                        except:
                            a = 1
                    else:
                        try:
                            if abs(float(a_list_strip[0])/float(a_list_strip[1]) - float(b_list_strip[0])/float(b_list_strip[1])) < 1e-6:
                                return True
                        except:
                            a = 1
                elif ",\!" in answer_b:
                    answer_a_strip = answer_a.replace(",\!", "").replace(" ","")
                    answer_b_strip = answer_b.replace(",\!", "").replace(" ","")
                    if answer_a_strip == answer_b_strip:
                        return True
                elif "pmatrix" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    a_list2, b_list2 = [], []
                    for x in a_list: 
                        a_list2 += x.split("/")
                    for x in b_list: 
                        b_list2 += x.split("/")
                    if a_list2 == b_list2 or "".join(a_list2) == "".join(b_list2):
                        return True
                elif "[" in answer_b and "]" in answer_b or "(" in answer_b and ")" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if a_list == b_list or "".join(a_list) == "".join(b_list):
                        return True
                elif any(keyword in answer_b for keyword in ["_2", "_4", "_6", "_8", "_10", "_16"]): 
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if a_list[0] == b_list[0]:
                        return True
                else:
                    replace_marker= ["\\left","\\right", " ", "$", "\\", r"\\", "(", ")", "{", "}", "[", "]", "\cup", "U"]  #  ["\\left","\\right", " ", "$", "\\", r"\\"]
                    answer_a = self.clean_answer(remove_boxes_keep_content(answer_a), replace_marker)
                    answer_b = self.clean_answer(answer_b, replace_marker)
                    answer_a = answer_a.split("=")[-1]
                    answer_b = answer_b.split("=")[-1]
                    if len(answer_a) > 10:
                        answer_a = answer_a.replace(".","").split("\n\n")[0].strip()
                        answer_b = answer_b.replace(".", "").strip()
                    # answer_a, answer_b = self.find_box(answer_a), self.find_box(answer_b)
                    if answer_a.lower().strip() == answer_b.lower().strip():
                        return True
                    else:
                        answer_a = answer_a.split("^")[0].strip()
                        answer_b = answer_b.split("^")[0].strip()
                        if answer_a.lower().strip() == answer_b.lower().strip():
                            return True
                    
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if set(a_list) == set(b_list):
                        return True
            except Exception as e:
                e = 0

        return res

    def extract_answer_from_gold_solution(self, solution: str):
        def remove_boxed(s):
            left = "\\boxed{"
            try:
                assert s[: len(left)] == left
                assert s[-1] == "}"
                return s[len(left) : -1]
            except:
                return None

        def last_boxed_only_string(string):
            idx = string.rfind("\\boxed")
            if idx < 0:
                idx = string.rfind("\\fbox")
                if idx < 0:
                    return None

            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx == None:
                retval = None
            else:
                retval = string[idx : right_brace_idx + 1]

            return retval

        return remove_boxed(last_boxed_only_string(solution))

    def extract_answer_from_model_completion(self, completion):
        answer_split = self.isolate_answer(completion)
        return answer_split


AIMEEvaluator = MATHEvaluator
AMCEvaluator = MATHEvaluator


class OdysseyMathEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
        
    def find_box(self, pred_str: str):
        ans = pred_str.split('boxed')[-1]
        if not ans:
            return ""    # origin:     return ""
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        return a

    # 替换指定字符的函数
    def clean_answer(self, answer, markers):
        for marker in markers:
            answer = answer.replace(marker, "")
        # 移除匹配的括号
        for open_bracket, close_bracket in [("(", ")"), ("[", "]"), ("{", "}")]:
            if open_bracket in answer and close_bracket in answer:
                answer = answer.replace(open_bracket, "").replace(close_bracket, "")
        return answer
    
    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        if answer_a == "" or answer_b == "":
            return False

        answer_a = answer_a.strip()
        answer_b = answer_b.strip()

        if answer_a.lower() == answer_b.lower():
            return True
        
        try:
            res = latex_equiv(answer_a, answer_b)
        except Exception as e:
            print(e)
            res = False
        
        # TODO: new_added
        if not res and answer_a != "\\":
            try:
                answer_a = answer_a.replace("$", "")
                answer_b = answer_b.replace("$", "")
                if "sqrt" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if a_list == b_list:
                        return True
                elif "frac" in answer_b or "/" in answer_b and "frac" in answer_a or "." in answer_b and "frac" in answer_a:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    a_list_strip, b_list_strip = [], []
                    for x in a_list:
                        a_list_strip += x.split("/")
                    for x in b_list:
                        b_list_strip += x.split("/")
                    if a_list_strip == b_list_strip or "".join(a_list_strip) == "".join(b_list_strip):
                        return True
                    elif len(a_list_strip) == 2 and len(b_list_strip) == 1:
                        if abs(float(a_list_strip[0])/float(a_list_strip[1]) - float(b_list_strip[0])) < 1e-6:
                            return True
                    elif len(a_list_strip) == 1 and len(b_list_strip) == 2:
                        if abs(float(b_list_strip[0])/float(b_list_strip[1]) - float(a_list_strip[0])) < 1e-6:
                            return True
                    else:
                        try:
                            if abs(float(a_list_strip[0])/float(a_list_strip[1]) - float(b_list_strip[0])/float(b_list_strip[1])) < 1e-6:
                                return True
                        except:
                            # print("Error")
                            a = 1
                elif ",\!" in answer_b:
                    answer_a_strip = answer_a.replace(",\!", "").replace(" ","")
                    answer_b_strip = answer_b.replace(",\!", "").replace(" ","")
                    if answer_a_strip == answer_b_strip:
                        return True
                elif "pmatrix" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    a_list2, b_list2 = [], []
                    for x in a_list: 
                        a_list2 += x.split("/")
                    for x in b_list: 
                        b_list2 += x.split("/")
                    if a_list2 == b_list2 or "".join(a_list2) == "".join(b_list2):
                        return True
                elif "[" in answer_b and "]" in answer_b or "(" in answer_b and ")" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if a_list == b_list or "".join(a_list) == "".join(b_list):
                        return True
                elif any(keyword in answer_b for keyword in ["_2", "_4", "_6", "_8", "_10", "_16"]): 
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if a_list[0] == b_list[0]:
                        return True
                elif answer_b in ["A", "B", "C", "D", "E"]:
                    if answer_b.lower()+")" in answer_a.lower() or answer_b.lower()+". " in answer_a.lower():
                        return True
                    elif  answer_b.lower() in re.findall(r'\\textbf{(.*?)}', answer_a)[0].lower():
                        return True
                else:
                    replace_marker= ["\\left","\\right", " ", "$", "\\", r"\\", "(", ")", "{", "}", "[", "]", "\cup", "U"]  #  ["\\left","\\right", " ", "$", "\\", r"\\"]
                    answer_a = self.clean_answer(remove_boxes_keep_content(answer_a), replace_marker)
                    answer_b = self.clean_answer(answer_b, replace_marker)
                    answer_a = answer_a.split("=")[-1]
                    answer_b = answer_b.split("=")[-1]
                    if len(answer_a) > 10:
                        answer_a = answer_a.replace(".","").split("\n\n")[0].strip()
                        answer_b = answer_b.replace(".", "").strip()
                    # answer_a, answer_b = self.find_box(answer_a), self.find_box(answer_b)
                    if answer_a.lower().strip() == answer_b.lower().strip():
                        return True
                    else:
                        answer_a = answer_a.split("^")[0].strip()
                        answer_b = answer_b.split("^")[0].strip()
                        if answer_a.lower().strip() == answer_b.lower().strip():
                            return True
                    
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if set(a_list) == set(b_list):
                        return True
            except Exception as e:
                e = 0

        return res

    def extract_answer_from_gold_solution(self, solution: str):
        def remove_boxed(s):
            left = "\\boxed{"
            try:
                assert s[: len(left)] == left
                assert s[-1] == "}"
                return s[len(left) : -1]
            except:
                return None

        def last_boxed_only_string(string):
            idx = string.rfind("\\boxed")
            if idx < 0:
                idx = string.rfind("\\fbox")
                if idx < 0:
                    return None

            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx == None:
                retval = None
            else:
                retval = string[idx : right_brace_idx + 1]

            return retval

        return remove_boxed(last_boxed_only_string(solution))

    def extract_answer_from_model_completion(self, completion):
        answer_split = self.isolate_answer(completion)
        return answer_split

MMHalBenchEvaluator = OdysseyMathEvaluator


class SVAMPEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False
        
        try:
            if abs(float(search_for_numbers(answer_a)[0]) - float(search_for_numbers(answer_b)[0])) < 1e-6:
                return True
        except:
            a = 1

        return correct

    def extract_answer_from_gold_solution(self, solution: str | float):
        """Extract the answer from the gold solution."""
        if isinstance(solution, float):
            return str(solution)
        return solution.strip()

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None

        assert isinstance(completion, str)

        preds = completion
        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return None
        else:
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]

        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]

        pred = pred.replace(",", "").replace("\n", "")
        is_number, pred = self._is_number(pred)
        if is_number:
            return pred
        else:
            return None


class STGEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _format_answer(self, answer: str):
        if answer.lower() in ["proved", "true", "yes", "correct", "positive", "affirmative", "right", "1", "t", "y"]:
            return "true"
        elif answer.lower() in ["disproved", "false", "no", "incorrect", "negative", "wrong", "0", "f", "n"]:
            return "false"
        else:
            return answer.lower()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        assert isinstance(answer_a, str) and isinstance(answer_b, str)

        format_answer_a = self._format_answer(answer_a)
        format_answer_b = self._format_answer(answer_b)
        return format_answer_a == format_answer_b or fuzz.token_sort_ratio(format_answer_a, format_answer_b) >= 90

    def extract_answer_from_gold_solution(self, solution: str):
        if solution is None:
            return None

        assert isinstance(solution, str)

        return self._format_answer(solution)

    def extract_answer_from_model_completion(self, completion: str):
        if completion is None:
            return None

        assert isinstance(completion, str)

        answer = self.isolate_answer(completion)
        if answer is None:
            return None

        return self._format_answer(answer)
    
class GPQAEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _format_answer(self, answer: str):
        if answer.strip() in ["A", "A:", "(A)", "(A", "A)", "(A):", "(A)", "0"]:
            return "A"
        elif answer.strip() in ["B", "B:", "(B)", "(B", "B)", "(B):", "(B)", "1"]:
            return "B"
        elif answer.strip() in ["C", "C:", "(C)", "(C", "C)", "(C):", "(C)", "2"]:
            return "C"
        elif answer.strip() in ["D", "D:", "(D)", "(D", "D)", "(D):", "(D)", "3"]:
            return "D"
        else:
            return answer.strip()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        assert isinstance(answer_a, str) and isinstance(answer_b, str)

        format_answer_a = self._format_answer(answer_a)
        format_answer_b = self._format_answer(answer_b)
        return format_answer_a == format_answer_b or fuzz.token_sort_ratio(format_answer_a, format_answer_b) >= 90

    def extract_answer_from_gold_solution(self, solution: str):
        if solution is None:
            return None

        assert isinstance(solution, str)

        return self._format_answer(solution)

    def extract_answer_from_model_completion(self, completion: str):
        if completion is None:
            return None

        assert isinstance(completion, str)

        answer = self.isolate_answer(completion)
        if answer is None:
            return None

        return self._format_answer(answer)


# MMVetEvaluator = MATHEvaluator

class MMStarEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _format_answer(self, answer: str):
        if answer.lower().strip() in ["a", "a:", "(a)", "(a", "a)", "(a):", "(a)", "0"]:
            return "a"
        elif answer.lower().strip() in ["b", "b:", "(b)", "(b", "b)", "(b):", "(b)", "1"]:
            return "b"
        elif answer.lower().strip() in ["c", "c:", "(c)", "(c", "c)", "(c):", "(c)", "2"]:
            return "c"
        elif answer.lower().strip() in ["d", "d:", "(d)", "(d", "d)", "(d):", "(d)", "3"]:
            return "d"
        else:
            return answer.strip()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        assert isinstance(answer_a, str) and isinstance(answer_b, str)

        format_answer_a = self._format_answer(answer_a)
        format_answer_b = self._format_answer(answer_b)
        return format_answer_a == format_answer_b or fuzz.token_sort_ratio(format_answer_a, format_answer_b) >= 90

    def extract_answer_from_gold_solution(self, solution: str):
        if solution is None:
            return None

        assert isinstance(solution, str)

        return self._format_answer(solution)

    def extract_answer_from_model_completion(self, completion: str):
        if completion is None:
            return None

        assert isinstance(completion, str)

        answer = self.isolate_answer(completion)
        if answer is None:
            return None

        return self._format_answer(answer)


class MathVistaEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
        
    def find_box(self, pred_str: str):
        ans = pred_str.split('boxed')[-1]
        if not ans:
            return ""    # origin:     return ""
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        return a

    # 替换指定字符的函数
    def clean_answer(self, answer, markers):
        for marker in markers:
            answer = answer.replace(marker, "")
        # 移除匹配的括号
        for open_bracket, close_bracket in [("(", ")"), ("[", "]"), ("{", "}")]:
            if open_bracket in answer and close_bracket in answer:
                answer = answer.replace(open_bracket, "").replace(close_bracket, "")
        return answer
    
    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        if answer_a == "" or answer_b == "":
            return False

        answer_a = answer_a.strip()
        answer_b = answer_b.strip()

        if answer_a.lower() == answer_b.lower():
            return True
        
        try:
            res = latex_equiv(answer_a, answer_b)
        except Exception as e:
            print(e)
            res = False
        
        # TODO: new_added
        if not res and answer_a != "\\":
            try:
                answer_a = answer_a.replace("$", "").replace("-", "").replace("*", "").strip()
                answer_b = answer_b.replace("$", "").replace("-", "").replace("*", "").strip()
                if "sqrt" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if a_list == b_list:
                        return True
                elif "frac" in answer_b or "/" in answer_b and "frac" in answer_a or "." in answer_b and "frac" in answer_a:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    a_list_strip, b_list_strip = [], []
                    for x in a_list:
                        a_list_strip += x.split("/")
                    for x in b_list:
                        b_list_strip += x.split("/")
                    if a_list_strip == b_list_strip or "".join(a_list_strip) == "".join(b_list_strip):
                        return True
                    elif len(a_list_strip) == 2 and len(b_list_strip) == 1:
                        if abs(float(a_list_strip[0])/float(a_list_strip[1]) - float(b_list_strip[0])) < 1e-6:
                            return True
                    elif len(a_list_strip) == 1 and len(b_list_strip) == 2:
                        if abs(float(b_list_strip[0])/float(b_list_strip[1]) - float(a_list_strip[0])) < 1e-6:
                            return True
                    else:
                        try:
                            if abs(float(a_list_strip[0])/float(a_list_strip[1]) - float(b_list_strip[0])/float(b_list_strip[1])) < 1e-6:
                                return True
                        except:
                            # print("Error")
                            a = 1
                elif ",\!" in answer_b:
                    answer_a_strip = answer_a.replace(",\!", "").replace(" ","")
                    answer_b_strip = answer_b.replace(",\!", "").replace(" ","")
                    if answer_a_strip == answer_b_strip:
                        return True
                elif "pmatrix" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    a_list2, b_list2 = [], []
                    for x in a_list: 
                        a_list2 += x.split("/")
                    for x in b_list: 
                        b_list2 += x.split("/")
                    if a_list2 == b_list2 or "".join(a_list2) == "".join(b_list2):
                        return True
                elif "[" in answer_b and "]" in answer_b or "(" in answer_b and ")" in answer_b:
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if a_list == b_list or "".join(a_list) == "".join(b_list):
                        return True
                elif any(keyword in answer_b for keyword in ["_2", "_4", "_6", "_8", "_10", "_16"]): 
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if a_list[0] == b_list[0]:
                        return True
                elif answer_b.lower() in ["a", "b", "c", "d", "e"]:
                    replace_marker= ["\\left","\\right", " ", "$", "\\", r"\\", "(", ")", "{", "}", "[", "]", "\cup", "U", "*"]
                    try:
                        try:
                            if answer_b.lower().strip() in answer_a.lower().split(" ")[0].strip():
                                return True
                        except:
                            a = 1
                        if answer_b.lower().strip() == answer_a.lower().strip():
                            return True
                        if answer_b.lower()+")" in answer_a.lower() and "ed)" not in answer_a.lower() or answer_b.lower()+". " in answer_a.lower():
                            return True
                        elif answer_b.lower() in self.clean_answer(remove_boxes_keep_content(answer_a), replace_marker).lower().strip() and len(self.clean_answer(remove_boxes_keep_content(answer_a), replace_marker).lower().strip()) <= 2:
                            return True
                        elif re.search(r"(?<=boxed\{).+?(?=\})", answer_a):
                            if answer_b.lower() in re.search(r"(?<=boxed\{).+?(?=\})", answer_a).group().lower().strip() and len(re.search(r"(?<=boxed\{).+?(?=\})", answer_a).group().lower().strip()) <= 2:
                                return True
                        elif  answer_b.lower() in re.findall(r'\\textbf{(.*?)}', answer_a)[0].lower():
                            return True
                    except:
                        a = 1
                elif len(search_for_numbers(answer_b)) == 1:
                    def convert_power_of_ten_to_float(s):
                        # 使用正则表达式匹配 "10^d" 形式的字符串
                        pattern = r"^10\^([-+]?\d+)$"
                        
                        match = re.match(pattern, s)
                        if match:
                            # 提取指数部分 d
                            exponent = int(match.group(1))
                            # 计算 10^d
                            return float(10 ** exponent)
                        else:
                            # 如果格式不匹配，返回 None 或者抛出异常
                            return None
                    try:
                        if abs(float(answer_a)-float(answer_b)) < 1e-5:
                            return True
                    except:
                        a = 1
                    
                    try:
                        if abs(float(convert_power_of_ten_to_float(answer_a))-float(answer_b)) < 1e-5:
                            return True
                    except:
                        a = 1
                    
                    try:
                        if "." not in search_for_numbers(answer_b)[0]:
                            answer_a_number = search_for_numbers(answer_a.split(".")[0].strip())[0]
                            answer_b_number = search_for_numbers(answer_b.strip())[0]
                            if abs(float(answer_a_number)-float(answer_b_number)) < 1e-5 or abs(10*float(answer_a_number)-float(answer_b_number)) < 1e-5 or abs(100*float(answer_a_number)-float(answer_b_number)) < 1e-5:
                                return True
                    except:
                        a = 1
                else:
                    replace_marker= ["\\left","\\right", " ", "$", "\\", r"\\", "(", ")", "{", "}", "[", "]", "\cup", "U", "\%", "%"]  #  ["\\left","\\right", " ", "$", "\\", r"\\"]
                    answer_a = self.clean_answer(remove_boxes_keep_content(answer_a), replace_marker)
                    answer_b = self.clean_answer(answer_b, replace_marker)
                    answer_a = answer_a.split("=")[-1]
                    answer_b = answer_b.split("=")[-1]
                    if len(answer_a) > 10:
                        answer_a = answer_a.replace(".","").split("\n\n")[0].strip()
                        answer_a = answer_a.split("is")[-1].strip() if len(answer_a.split("is")[-1].strip())!=0 else answer_a
                        answer_b = answer_b.replace(".", "").strip()
                    # answer_a, answer_b = self.find_box(answer_a), self.find_box(answer_b)
                    if answer_a.lower().strip() == answer_b.lower().strip():
                        return True
                    else:
                        answer_a = answer_a.split("^")[0].strip()
                        answer_b = answer_b.split("^")[0].strip()
                        if answer_a.lower().strip() == answer_b.lower().strip():
                            return True
                    
                    a_list = search_for_numbers(answer_a)
                    b_list = search_for_numbers(answer_b)
                    if set(a_list) == set(b_list):
                        return True
            except Exception as e:
                e = 0

        return res

    def extract_answer_from_gold_solution(self, solution: str):
        def remove_boxed(s):
            left = "\\boxed{"
            try:
                assert s[: len(left)] == left
                assert s[-1] == "}"
                return s[len(left) : -1]
            except:
                return None

        def last_boxed_only_string(string):
            idx = string.rfind("\\boxed")
            if idx < 0:
                idx = string.rfind("\\fbox")
                if idx < 0:
                    return None

            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx == None:
                retval = None
            else:
                retval = string[idx : right_brace_idx + 1]

            return retval

        return remove_boxed(last_boxed_only_string(solution))

    def extract_answer_from_model_completion(self, completion):
        answer_split = self.isolate_answer(completion)
        if answer_split in ["", "(", ")", ",", ":"]:   # TODO: 01-10 14：23 新加的代码块，这里一会要测试一下加了之后之前的测试代码是否有影响
            answer_split = self.isolate_answer(completion[:-8])
        
        try:
            if "answer is:" not in completion and "the correct choice is" in completion:
                answer_split = completion.split("the correct choice is")[-1].replace(":","").strip()
            elif "answer is:" not in completion and "The correct choice is" in completion:
                answer_split = completion.split("The correct choice is")[-1].replace(":","").strip()
            elif "answer is:" not in completion and "final answer:" in completion:
                answer_split = completion.split("final answer: ")[-1].replace(":","").strip()
            elif "answer is:" not in completion and "Final answer:" in completion:
                answer_split = completion.split("Final answer: ")[-1].replace(":","").strip()   # "the correct answer is "
            elif "answer is:" not in completion and "the correct answer is:" in completion:
                answer_split = completion.split("the correct answer is:")[-1].replace(":","").strip()
            elif "answer is:" not in completion and "The correct answer is:" in completion:
                answer_split = completion.split("The correct answer is:")[-1].replace(":","").strip()             
        except:
            a = 1
        return answer_split

MathVerseEvaluator = MathVistaEvaluator
MathVisionEvaluator = MathVistaEvaluator
GAOKAOMMEvaluator = MathVistaEvaluator
ChartQAEvaluator = MathVistaEvaluator
WeMathEvaluator = MathVistaEvaluator

MathVistaTextEvaluator = MathVistaEvaluator

DynaMathEvaluator = MathVistaEvaluator

MMVetEvaluator = MathVistaEvaluator
