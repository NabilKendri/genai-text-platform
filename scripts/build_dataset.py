from pathlib import Path
from datasets import load_dataset

#The script downloads 50000 labeled new articles from Hugging Face and saves each one as its own .txt file in data/raw.


#The location of the folder where the raw data will be stored in the object OUT.
OUT = Path("data/raw")
#OUT.mkdir() -> makes directory, creates the folder in the destination stored into the OUT object.
    #Create the parent folders if missing, if folder already exists then continue.
OUT.mkdir(parents=True, exist_ok =True)
    #each word here is mapped to a number, through its position in the list.
    #in the dataset, the label is already stored into category number.
    #LABELS will turn that number into a readable word in the filename.
    #Later, we'll check if our AI system guesses the right category by comparing its awnser to the one in the filename.
        #This is ou awnser key to measure accuracy.
LABELS = ["world", "sports", "business", "scitech"]

#The dataset is loaded from the hugging face
    #it's divided into 2 parts (splits) : 1- train and test.
ds = load_dataset("fancyzhx/ag_news", split="train").select(range(50_000))
    #create .txt file per article, 50000 times.
for i, row in enumerate(ds):
    (OUT / f"doc_{i:05d}_{LABELS[row['label']]}.txt").write_text(row["text"], encoding="utf-8")


#Expected output : Wrote 50000 files.
print(f"Wrote {len(list(OUT.glob('*.txt')))} files")