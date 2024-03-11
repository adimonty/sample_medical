import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import scipy.spatial as sp
import pandas as pd
from docx import Document
import os
import matplotlib.pyplot as plt
import shutil
import tempfile

# Load the pre-trained sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

def detect_titles(pdf_filename, model):
    titles = []
    with fitz.open(pdf_filename) as pdf:
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text_blocks = page.get_text("blocks")
            medical_concept_embedding = model.encode("medical plan comparison")
            for block in text_blocks:
                block_text = " ".join(block[4].split())
                if block_text:
                    embedding = model.encode(block_text)
                    similarity = 1 - sp.distance.cosine(embedding, medical_concept_embedding)
                    if similarity >= 0.65:
                        titles.append((page_num, block_text))
    return titles

def find_table_locations(pdf_path, titles):
    """
    Find the approximate locations of tables on PDF pages based on titles.

    Args:
        pdf_path (str): Path to the PDF file.
        titles (list): A list of titles preceding the tables.

    Returns:
        dict: A dictionary with page numbers as keys and table bounding box (fitz.Rect) as values.
    """
    table_locations = {}
    doc = fitz.open(pdf_path)

    for title in titles:
        for page in doc:
            text_instances = page.search_for(title)
            for inst in text_instances:
                # Assuming table immediately follows title; adjust y1 as needed
                # These coordinates might need adjustment based on actual document layout
                table_rect = fitz.Rect(inst.x0, inst.y1, inst.x1, inst.y1 + 300)  # Example rect; adjust as needed
                table_locations[page.number] = table_rect
                break  # Assuming only one occurrence per page

    doc.close()
    return table_locations


def docx_to_dataframes(docx_filename):
    doc = Document(docx_filename)
    tables = []
    for table in doc.tables:
        data = [[cell.text for cell in row.cells] for row in table.rows]
        df = pd.DataFrame(data[1:], columns=data[0])  # Use the first row as headers
        tables.append(df)
    return tables

def dataframe_to_image(df, image_path):
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Adjust as needed
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def insert_image_into_pdf(pdf_path, image_path, page_number, image_rect):
    # Create a temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
    os.close(temp_fd)  # Close the file descriptor
    
    # Copy the original PDF to the temporary file
    shutil.copyfile(pdf_path, temp_path)

    # Open the temporary file for modifications
    doc = fitz.open(temp_path)
    page = doc.load_page(page_number)
    page.insert_image(image_rect, filename=image_path)
    doc.close()  # Close the document to ensure all changes are written

    # Define the final output file name
    output_pdf_path = pdf_path.replace(".pdf", "_modified.pdf")

    # Move the temporary file to the desired output location
    shutil.move(temp_path, output_pdf_path)


def main():
    pdf_filename = 'NOV_Highlights-Guide_2024.pdf'
    docx_filename = 'Quote Table - input.docx'
    output_folder = '.'

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    titles = [title_text for _, title_text in detect_titles(pdf_filename, model)]  # Extract titles from detection function
    table_locations = find_table_locations(pdf_filename, titles)
    docx_tables = docx_to_dataframes(docx_filename)

    for i, (page_num, rect) in enumerate(table_locations.items()):
        if i < len(docx_tables):
            df = docx_tables[i]
            image_path = os.path.join(output_folder, f"table_{i}.png")
            dataframe_to_image(df, image_path)

            # Use detected table locations for inserting images
            insert_image_into_pdf(pdf_filename, image_path, page_num, rect)

if __name__ == "__main__":
    main()
