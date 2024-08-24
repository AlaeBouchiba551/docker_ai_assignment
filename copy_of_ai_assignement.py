import fitz
import os
import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load the PDF
pdf_path = '2307.06435v9.pdf'
pdf_document = fitz.open(pdf_path)

# Create directories to save images that exist in the pdf
if not os.path.exists('extracted_images'):
    os.makedirs('extracted_images')

# Function to extract images and text from each page of the pdf
def extract_text_and_images(pdf_document):
    text_chunks = []
    images = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # Extract text
        text = page.get_text("text")
        text_chunks.append(text)

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"page_{page_num+1}_img_{img_index}.{image_ext}"
            image_path = os.path.join('extracted_images', image_name)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            images.append(image_path)

    return text_chunks, images

# Run the extraction
text_chunks, images = extract_text_and_images(pdf_document)

# Function to split text into chunks
def split_text_into_chunks(text_chunks, max_chunk_size=1000):
    split_chunks = []

    for text in text_chunks:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunk = ""
        for sentence in sentences:
            if len(chunk) + len(sentence) <= max_chunk_size:
                chunk += sentence + " "
            else:
                if chunk.strip():  # Only add non-empty chunks
                    split_chunks.append(chunk.strip())
                chunk = sentence + " "
        if chunk.strip():  # Add the last chunk if not empty
            split_chunks.append(chunk.strip())

    return split_chunks

# Split the text into chunks
text_chunks_split = split_text_into_chunks(text_chunks)

# Load a pre-trained CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate text embeddings using CLIP
def generate_clip_text_embeddings(texts, clip_model, clip_processor, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = clip_processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            batch_embeddings = clip_model.get_text_features(**inputs).cpu()
        embeddings.append(batch_embeddings)
    
    if embeddings:
        return torch.cat(embeddings)
    else:
        return torch.tensor([])  # Return an empty tensor if no embeddings

# Generate text embeddings using CLIP
text_embeddings_clip = generate_clip_text_embeddings(text_chunks_split, clip_model, clip_processor)
print("Shape of text embeddings:", text_embeddings_clip.shape)

# Function to generate image embeddings in batches
def generate_image_embeddings_in_batches(images, clip_model, clip_processor, batch_size=8):
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_embeddings = []
        for image_path in batch:
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue
            
            try:
                inputs = clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    embedding = clip_model.get_image_features(**inputs).cpu()
                batch_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        if batch_embeddings:
            embeddings.append(torch.cat(batch_embeddings))
        else:
            print(f"No valid embeddings for batch {i // batch_size + 1}.")
    
    if embeddings:
        return torch.cat(embeddings)
    else:
        return torch.tensor([])  # Return an empty tensor if no embeddings

# Generate image embeddings using CLIP
image_embeddings_clip = generate_image_embeddings_in_batches(images, clip_model, clip_processor)
print("Shape of image embeddings:", image_embeddings_clip.shape)

# Function to perform search based on the query
def search(query, text_embeddings, image_embeddings, text_chunks, images, search_type="both"):
    # Generate embeddings for the query
    query_embedding = generate_clip_text_embeddings([query], clip_model, clip_processor)
    
    if query_embedding.numel() == 0:
        print("Query embedding is empty.")
        return [], []

    # Initialize empty lists for results
    top_text_chunks = []
    top_images = []

    # Perform search based on the search_type parameter
    if search_type in ["text", "both"]:
        # Compute cosine similarity for text embeddings
        if text_embeddings.numel() == 0:
            print("No text embeddings available for search.")
            return [], top_images
        text_similarities = cosine_similarity(query_embedding.numpy(), text_embeddings.numpy())
        text_similarities = text_similarities.flatten()

        # Find the top 3 most similar text chunks
        top_text_indices = np.argsort(-text_similarities)[:3]
        top_text_chunks = [(text_chunks[i], text_similarities[i]) for i in top_text_indices]

    if search_type in ["image", "both"]:
        # Compute cosine similarity for image embeddings
        if image_embeddings.numel() == 0:
            print("No image embeddings available for search.")
            return top_text_chunks, []
        image_similarities = cosine_similarity(query_embedding.numpy(), image_embeddings.numpy())
        image_similarities = image_similarities.flatten()

        # Find the top 3 most similar images
        top_image_indices = np.argsort(-image_similarities)[:3]
        top_images = [(images[i], image_similarities[i]) for i in top_image_indices]

    return top_text_chunks, top_images

# User query
query = "Large Language Model"
search_type = "both"  # Can be "text", "image", or "both"

# Perform the search
top_text_chunks, top_images = search(query, text_embeddings_clip, image_embeddings_clip, text_chunks_split, images, search_type=search_type)

# Function to display images with their scores
def display_images_with_scores(image_paths, scores):
    if not image_paths:
        print("No images to display.")
        return
    
    for img_path, score in zip(image_paths, scores):
        try:
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.title(f"Score: {score:.4f}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error displaying image {img_path}: {e}")

# Display the results
if search_type in ["text", "both"]:
    print("Top 3 Text Chunks:")
    for chunk, score in top_text_chunks:
        print(f"Score: {score:.4f} - Text: {chunk}\n")

if search_type in ["image", "both"]:
    print("\nTop 3 Images:")
    display_images_with_scores(
        [img_path for img_path, _ in top_images],  # Extract paths
        [score for _, score in top_images]         # Extract scores
    )
!pip install transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
/////////////////////////////////////////////////////////// RAG part//////////////////////////////////////////////////////////////////
# Function to generate an answer using retrieved chunks
def generate_answer(query, retrieved_chunks):
    # Concatenate the retrieved chunks into a single context
    context = " ".join([chunk for chunk, _ in retrieved_chunks])

    # Encode the input (query + context)
    inputs = tokenizer(query + " " + context, return_tensors="pt", max_length=1024, truncation=True)

    # Generate a response
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=5, early_stopping=True)

    # Decode the response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

# Load the generative model (BART in this case)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# User query
query = "Fine Tuning"
search_type = "both"  # Can be "text", "image", or "both"

# Perform the search to get top text chunks
top_text_chunks, _ = search(query, text_embeddings_clip, image_embeddings_clip, text_chunks_split, images, search_type=search_type)

# Generate an answer using the retrieved text chunks
answer = generate_answer(query, top_text_chunks)

print("Generated Answer:")
print(answer)
