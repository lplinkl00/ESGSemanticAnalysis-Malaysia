import fitz #PymuPDF
import json

def pdf_to_json(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Initialize a dictionary to hold the extracted content
    pdf_dict = {"pages": []}
    
    # Extract text from each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        pdf_dict["pages"].append({"page_number": page_num + 1, "content": text})
    
    # Convert the dictionary to JSON
    json_data = json.dumps(pdf_dict, indent=4)
    
    # Optionally, save the JSON data to a file
    with open(pdf_path.replace('.pdf', '.json'), 'w') as json_file:
        json_file.write(json_data)
    
    return json_data

# Replace 'your_pdf_file.pdf' with the path to your PDF file
pdf_to_json('PDF2JSON\GRI 3_ Material Topics 2021.pdf')
