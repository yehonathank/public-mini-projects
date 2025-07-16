import fitz  # from PyMuPDF
import os

class PDF_Scraper:
    """
    PDF_Scraper(pdf_path, save_directory, new_filename).run()
    - Extracts text from PDF and saves to .txt file in given directory.
    """
    def __init__(self, pdf_path: str, save_directory: str, new_filename: str):
        self.pdf_path = pdf_path
        self.save_directory = save_directory
        self.new_filename = new_filename if new_filename.endswith('.txt') else new_filename + '.txt'

    def extract_text(self) -> str:
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()

    def save_text_to_dir(self, text: str):
        os.makedirs(self.save_directory, exist_ok=True)
        save_path = os.path.join(self.save_directory, self.new_filename)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {save_path}")

    def run(self):
        text = self.extract_text()
        self.save_text_to_dir(text)

def test():
    scraper = PDF_Scraper("llm_projects/rag/docs/Benjamin_Netanyahu.pdf", "llm_projects/document_scraper/scraped_docs", "bibi.txt")
    scraper.run()

if __name__ == "__main__":
    test()