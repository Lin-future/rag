import nltk

def download_nltk_resources():
    resources = ["punkt", "averaged_perceptron_tagger"]
    for resource in resources:
        try:
            print(f"Attempting to download NLTK resource: {resource}")
            nltk.download(resource, quiet=True) # quiet=True 减少输出
            print(f"Successfully downloaded/updated {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
            print(f"  If you encounter network issues, you might need to configure a proxy or try again later.")
            print(f"  Alternatively, try in a Python interpreter: import nltk; nltk.download(\'{resource}\')")

if __name__ == "__main__":
    download_nltk_resources()