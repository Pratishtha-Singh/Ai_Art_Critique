"""
This module contains data processing nodes for the AI Art Critic project.
"""

import pandas as pd
import re
import logging
import torch
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image
import requests

logger = logging.getLogger(__name__)


def clean_html_text(text: str) -> str:
    """
    Remove HTML tags and clean text.
    
    Args:
        text: Input text that may contain HTML
        
    Returns:
        Cleaned text without HTML tags
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove HTML tags using regex
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Decode common HTML entities
    clean_text = clean_text.replace('&amp;', '&')
    clean_text = clean_text.replace('&lt;', '<')
    clean_text = clean_text.replace('&gt;', '>')
    clean_text = clean_text.replace('&quot;', '"')
    clean_text = clean_text.replace('&#39;', "'")
    
    # Normalize whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text


def normalize_text(text: str) -> str:
    """
    Normalize casing for titles and names.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text with proper casing
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.strip()


def merge_datasets(
    wikiart: pd.DataFrame, 
    artemis: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge WikiArt and ArtEmis datasets into unified schema.
    
    The unified schema includes:
    [id, artist, title, year, style, genre, image_url, emotion_text, dataset_source]
    
    Args:
        wikiart: WikiArt dataset with art metadata
        artemis: ArtEmis dataset with emotion annotations
        
    Returns:
        Unified dataset with both sources combined
    """
    logger.info("Starting dataset merge process")
    logger.info(f"WikiArt entries: {len(wikiart)}")
    logger.info(f"ArtEmis entries: {len(artemis)}")
    
    # Step 1: Prepare data for merging - extract base filename from WikiArt
    wikiart_copy = wikiart.copy()
    artemis_copy = artemis.copy()
    
    # Extract base filename from WikiArt (remove path and extension)
    def extract_base_filename(filename):
        if pd.isna(filename):
            return ""
        # Remove directory path
        base = str(filename).split('/')[-1]
        # Remove file extension
        base = base.split('.')[0] if '.' in base else base
        return base
    
    wikiart_copy['base_filename'] = wikiart_copy['filename'].apply(extract_base_filename)
    
    logger.info("Sample WikiArt base filenames:")
    logger.info(wikiart_copy['base_filename'].head().tolist())
    logger.info("Sample ArtEmis paintings:")
    logger.info(artemis_copy['painting'].head().tolist())
    
    # Step 2: Merge WikiArt and ArtEmis on base filename
    merged_df = pd.merge(
        wikiart_copy,
        artemis_copy,
        how="left",
        left_on="base_filename",
        right_on="painting",
        suffixes=('_wiki', '_artemis')
    )
    
    logger.info(f"WikiArt + ArtEmis merged: {len(merged_df)} entries")
    
    # Step 2: Handle duplicate columns from merge
    # Keep WikiArt versions for metadata, ArtEmis for emotion data
    columns_to_drop = []
    for col in merged_df.columns:
        if col.endswith('_artemis') and col.replace('_artemis', '_wiki') in merged_df.columns:
            # If it's not emotion-related, drop artemis version
            if 'emotion' not in col.lower() and 'utterance' not in col.lower():
                columns_to_drop.append(col)
    
    merged_df = merged_df.drop(columns=columns_to_drop)
    
    # Rename remaining _wiki columns to remove suffix
    rename_dict = {col: col.replace('_wiki', '') for col in merged_df.columns if col.endswith('_wiki')}
    merged_df = merged_df.rename(columns=rename_dict)
    
    # Step 3: Add dataset source
    merged_df['dataset_source'] = 'WikiArt_ArtEmis'
    
    # Step 4: Map emotion text from ArtEmis utterance column
    if 'utterance' in merged_df.columns:
        merged_df['emotion_text'] = merged_df['utterance']
    else:
        merged_df['emotion_text'] = ""
    
    # Step 5: Map columns to unified schema
    column_mapping = {
        'description': 'title',  # WikiArt description becomes title
        'art_style': 'style',    # ArtEmis art_style
        'genre': 'genre',        # Keep genre
        'filename': 'image_name' # Keep filename as image_name
    }
    
    merged_df = merged_df.rename(columns=column_mapping)
    
    # Step 6: Ensure all required columns exist
    required_columns = [
        'artist', 'title', 'year', 'style', 'genre', 
        'image_name', 'emotion_text', 'dataset_source'
    ]
    
    for col in required_columns:
        if col not in merged_df.columns:
            merged_df[col] = ""
    
    # Step 6: Apply text cleaning and standardization
    logger.info("Applying text cleaning and standardization")
    
    # Clean HTML from text fields
    text_columns = ['artist', 'title', 'emotion_text']
    for col in text_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].apply(clean_html_text)
    
    # Normalize artist and title names
    merged_df['artist'] = merged_df['artist'].apply(normalize_text)
    merged_df['title'] = merged_df['title'].apply(normalize_text)
    
    # Step 7: Handle missing values
    merged_df = merged_df.fillna({
        'emotion_text': '',
        'style': 'Unknown',
        'genre': 'Unknown',
        'artist': 'Unknown Artist',
        'title': 'Untitled'
    })
    
    # Step 8: Clean and standardize year column
    # Extract year from strings like "1890s" or "ca. 1900"
    def extract_year(year_value):
        if pd.isna(year_value):
            return None
        year_str = str(year_value)
        # Try to extract 4-digit year
        match = re.search(r'\b(1\d{3}|20\d{2})\b', year_str)
        if match:
            return int(match.group(1))
        return None
    
    if 'year' in merged_df.columns:
        merged_df['year'] = merged_df['year'].apply(extract_year)
    
    # Step 9: Generate unique IDs
    merged_df.insert(0, 'id', range(1, len(merged_df) + 1))
    
    # Step 10: Final validation
    logger.info(f"Final unified dataset: {len(merged_df)} entries")
    logger.info(f"Columns: {list(merged_df.columns)}")
    logger.info(f"Entries with emotion_text: {(merged_df['emotion_text'] != '').sum()}")
    
    return merged_df


def generate_text_embeddings(unified_df: pd.DataFrame, sample_size: int = 10000) -> pd.DataFrame:
    """
    Generate text embeddings for the unified dataset using BERT-tiny.
    
    Args:
        unified_df: DataFrame with unified art data
        sample_size: Number of samples to process (None for all)
        
    Returns:
        DataFrame with added text_embedding column
    """
    if sample_size:
        unified_df = unified_df.head(sample_size).copy()
        logger.info(f"Processing subset of {sample_size} samples")
    
    logger.info("Loading text model and tokenizer...")
    text_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    text_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    embeddings = []
    
    for idx, row in unified_df.iterrows():
        text = f"{row.get('title', '')} {row.get('utterance', '')} {row.get('emotion_text', '')}".strip()
        if not text:
            text = "No description available"
        
        inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embedding = text_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
        
        embeddings.append(embedding)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1} embeddings")
    
    unified_df = unified_df.copy()
    unified_df['text_embedding'] = embeddings
    
    logger.info(f"Generated {len(embeddings)} text embeddings")
    return unified_df


def generate_image_embeddings(unified_df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """
    Generate image embeddings for the unified dataset using CLIP.
    Selects artworks with high repetition (≥5 emotion annotations), diverse styles, and many artists.
    
    Args:
        unified_df: DataFrame with unified art data (from text embeddings)
        sample_size: Number of samples to process
        
    Returns:
        DataFrame with added image_embedding column, filtered to sample_size diverse artworks
    """
    # Filter for artworks with ≥5 emotion annotations
    repetition_counts = unified_df.groupby('image_name').size()
    high_rep_artworks = repetition_counts[repetition_counts >= 5].index
    filtered_df = unified_df[unified_df['image_name'].isin(high_rep_artworks)]
    
    # Select diverse: group by style and artist, take top
    # To ensure diversity, sort by style, then artist, take first sample_size
    filtered_df = filtered_df.sort_values(['style', 'artist']).drop_duplicates(subset=['image_name']).head(sample_size)
    
    logger.info(f"Selected {len(filtered_df)} diverse artworks with ≥5 repetitions")
    
    logger.info("Loading CLIP model and processor...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    embeddings = []
    successful_rows = []
    
    for idx, row in filtered_df.iterrows():
        image_name = row.get('image_name', '')
        if image_name:
            # Extract artist and title from image_name
            parts = image_name.split('/')
            if len(parts) == 2:
                artist_title = parts[1].replace('_', '/').replace('.jpg', '')
                artist, title = artist_title.split('/', 1)
                image_url = f"https://uploads6.wikiart.org/images/{artist}/{title}.jpg"
            else:
                image_url = f"https://uploads6.wikiart.org/images/{image_name}"
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            try:
                response = requests.get(image_url, stream=True, headers=headers, timeout=10)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                    inputs = clip_processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        embedding = clip_model.get_image_features(**inputs).squeeze().numpy()
                    embeddings.append(embedding)
                    successful_rows.append(row)
                    if len(successful_rows) % 10 == 0:
                        logger.info(f"Successfully processed {len(successful_rows)} image embeddings")
                else:
                    logger.debug(f"Failed download for {image_name}: HTTP {response.status_code}")
            except Exception as e:
                logger.debug(f"Failed to download image for {image_name}: {e}")
        else:
            logger.debug(f"No image_name for row {idx}")
    
    # Create new df with only successful rows
    unified_df = pd.DataFrame(successful_rows).copy()
    unified_df['image_embedding'] = embeddings
    
    logger.info(f"Generated {len(embeddings)} image embeddings (skipped failed downloads)")
    return unified_df