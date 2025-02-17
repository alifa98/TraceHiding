from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# DATASET_NAME = "HO_Rome_Res8"
# DATASET_NAME = "HO_Porto_Res8"
# DATASET_NAME = "HO_Geolife_Res8"
DATASET_NAME = "HO_NYC_Res9"

# Initialize the tokenizer with an empty vocabulary
tokenizer = Tokenizer(WordLevel(vocab={}, unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

# Trainer to learn the vocabulary
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"])
tokenizer.train(files=[f"datasets/plain_sequence_corp_{DATASET_NAME}.txt"], trainer=trainer)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]"
)

# Save the tokenizer for future use
hf_tokenizer.save_pretrained(f"experiments/{DATASET_NAME}/saved_models/ho-sequence-tokenizer")