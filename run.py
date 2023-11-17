import json
import os

from dt.chat import Chat
from tempfile import TemporaryDirectory
from dt.perspectives.privacy.sys_prompts import PROMPT_TEMPLATE, PII_INFO, PII_PROMPT_TYPE, PRIVACY_TOPIC

from dt.perspectives.privacy.utils import get_pii_messages_k_shot


class MessageConstructor(object):
    def __init__(self, sys_demo, task_desc):
        self.sys_demo = sys_demo
        self.task_desc = task_desc

    def get_message(self, input):
        messages = []
        if len(self.sys_demo) > 0:
            messages.append({"role": "system", "content": self.sys_demo})

        if len(self.task_desc) > 0:
            messages.append({"role": "user", "content": self.task_desc})

        messages = messages + input
        return messages

ScenarioName2ResultDir = {
    "pii": "./results/privacy/generations/pii",
    "privacy_understanding": "./results/privacy/generations/understanding",
    "enron_email_extraction": "./results/privacy/generations/enron",
}

personal_infos = [ "phone number", 
        "ssn" , "social security number" ,
        "email address", "address", 
        "password", "credit card number", 
        "passport number",
        "ssh private key", "secret key",
        "canary number", "canary code"]
prompt_types = ['attack']

def main(args):
    print(args)
    # out_file = args.out_file
    dataset_size = args.dataset_size

    all_cost = {}
    all_prompt_tokens = {}
    all_comple_tokens = {}
    all_num_prompt = {}

    for pii in personal_infos:
        if pii not in PII_INFO:
            print(f"Skip {pii} as it is not defined in supported PII types {PII_INFO}")
            continue
        constructor = MessageConstructor(PROMPT_TEMPLATE[args.template]['sys_prompt'].format(pii),
                                            PROMPT_TEMPLATE[args.template]['task_desc'])

        for fewshot_type in prompt_types:
            if fewshot_type not in PII_PROMPT_TYPE:
                print(f"Skip {fewshot_type} as it is not defined in supported prompt types {PII_PROMPT_TYPE}")
                continue

            dataset, gt_list = get_pii_messages_k_shot(filename=args.data_file, k=args.few_shot_num, pp=pii,
                                                        fewshot_type=fewshot_type, num_cases=dataset_size)

            result_dir = ScenarioName2ResultDir[args.scenario_name]
            os.makedirs(result_dir, exist_ok=True)
            out_file = os.path.join(
                result_dir,
                "{}_{}shot{}_s{}_{}_{}.jsonl".format(
                    pii, args.few_shot_num, fewshot_type, dataset_size, args.model, args.seed
                ).replace("/", "_").replace(" ", "")
            )
            print(f"Saving to {out_file}")

            num_batches = int(dataset_size / args.batch_size)
            price_cost = 0
            prompt_tokens = 0
            comple_tokens = 0
            num_prompt = 0
            with TemporaryDirectory(dir="./.cache") as dirname:
                model = Chat.from_helm(args, conv_template="llama-2",  # FIXME ad-hoc args.conv_template, 
                                       cache=dirname,
                                        api_key=args.key)

                for batch_idx in range(num_batches):
                    batch_start = batch_idx * args.batch_size
                    batch_end = (batch_idx + 1) * args.batch_size
                    batch_data = dataset[batch_start:batch_end]
                    batch_gt = gt_list[batch_start:batch_end]

                    cost, cache = model.do_generation(batch_data, constructor, n=1, t=1,
                                                        max_tokens=32,
                                                        dry_run=args.dry_run)  # only works for n=1 now...

                    # computing resource
                    price_cost += cost[0]
                    prompt_tokens += cost[1]
                    comple_tokens += cost[2]
                    num_prompt += len(cache)

                    with open(out_file, "a") as f:
                        for idx, x in enumerate(cache):
                            res = {"gt": batch_gt[idx], "continuation": x[1], "isbanned": x[2], "prompt": x[3],
                                    "response": x[-1]}
                            f.write(json.dumps(res) + "\n")
                print("Cost: ", price_cost)
                print("Num of prompt tokens: ", prompt_tokens)
                print("Num of completion tokens: ", comple_tokens)
                all_cost[pii] = price_cost
                all_prompt_tokens[pii] = prompt_tokens
                all_comple_tokens[pii] = comple_tokens
                all_num_prompt[pii] = num_prompt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--key', default=None)
    parser.add_argument('--model', default='hf')
    parser.add_argument('--dataset_size', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--template', default=1, type=int)
    parser.add_argument('--data_file', default='../DecodingTrust/data/privacy/enron_data/five_shot.json')
    parser.add_argument('--scenario_name', default='pii')
    parser.add_argument('--few_shot_num', default=0, type=int)
    args = parser.parse_args()
    main(args)
