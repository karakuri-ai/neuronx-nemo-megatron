from itertools import chain

from datasets import load_dataset, Features, Sequence, Value
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
block_size = 2048
features = Features({
    "tokens": Sequence(feature=Value(dtype="int32")),
    "labels": Sequence(feature=Value(dtype="int32")),
    "loss_mask": Sequence(feature=Value(dtype="float32")),
})

BOS = tokenizer.bos_token_id
EOS = tokenizer.eos_token_id
BINST = tokenizer.encode("[INST]", add_special_tokens=False)
EINST = tokenizer.encode("[/INST]", add_special_tokens=False)


def tokenize(example):
    tokens = [BOS]
    labels = [0]
    loss_mask = [0.0]

    if example["messages"][0]["role"] == "system":
        system = example["messages"][0]["content"]
        msgs = example["messages"][1:]
    else:
        system = None
        msgs = example["messages"]

    for i in range(0, len(msgs) // 2 * 2, 2):
        assert msgs[i]["role"] == "user"
        assert msgs[i + 1]["role"] == "assistant"

        usr = msgs[i]["content"]
        if i == 0 and system is not None:
            usr = "<<SYS>>\n" + system + "\n<</SYS>>\n\n" + usr
        asst = msgs[i + 1]["content"]

        inputs = BINST + tokenizer.encode(usr, add_special_tokens=False) + EINST
        outputs = tokenizer.encode(asst, add_special_tokens=False) + [EOS]

        tokens += inputs + outputs
        labels += [0] * (len(inputs) - 1) + outputs + [0]
        loss_mask += [0.0] * (len(inputs) - 1) + [1.0] * len(outputs) + [0.0]

    return {"tokens": tokens, "labels": labels, "loss_mask": loss_mask}


def group_texts(examples):
    concatenated_examples = {
        k: list(chain.from_iterable(values for values in examples[k]))
        for k in features.keys()
    }
    total_length = len(concatenated_examples["tokens"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def main():
    ds = load_dataset("HuggingFaceH4/no_robots")
    ds = (
        ds
        .shuffle(seed=42)
        .map(
            tokenize,
            remove_columns=ds["train"].column_names,
            features=features,
        )
        .map(
            group_texts,
            batched=True,
            batch_size=10000,
        )
        .filter(lambda example: sum(example["loss_mask"]) > 0)
        .map(lambda _: {"attention_mask": True, "position_ids": True})
    )
    ds["validation"] = ds.pop("test")
    ds.save_to_disk("data/mixtral_sft")


if __name__ == "__main__":
    main()
