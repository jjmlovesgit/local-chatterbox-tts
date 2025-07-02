# sentence_chunker.py
from typing import List, Generator
import logging

# Assuming TextSegmenter is in text_segmenter.py as per your setup
from text_segmenter import TextSegmenter 

logger = logging.getLogger(__name__)

class SentenceChunkerForTTS:
    """
    Processes a block of text, identifies sentences using TextSegmenter,
    and groups them into chunks of a specified size for TTS. (Python Version)
    """
    def __init__(self, sentences_per_chunk: int = 1, language: str = "en"):
        if sentences_per_chunk < 1:
            raise ValueError("sentences_per_chunk must be at least 1.")
        self.text_segmenter = TextSegmenter(language=language)
        self.sentences_per_chunk: int = sentences_per_chunk
        logger.debug(f"SentenceChunkerForTTS initialized with {sentences_per_chunk} sentences per chunk for language '{language}'.")

    def chunk_text(self, full_text: str) -> Generator[str, None, None]:
        """
        Takes a full block of text, segments it into sentences,
        and yields N-sentence chunks.

        Args:
            full_text (str): The entire text to be chunked.

        Yields:
            str: A string containing `self.sentences_per_chunk` sentences joined,
                 or the remaining sentences if less than a full chunk at the end.
        """
        if not full_text or not full_text.strip():
            logger.debug("chunk_text called with empty or whitespace-only text.")
            return

        self.text_segmenter.clear_buffer() 
        
        processed_text = full_text.strip()
        # Add a period to help pysbd finalize the last sentence if it's not terminated
        # and if there's actual content.
        if processed_text and not processed_text.endswith(tuple(self.text_segmenter.terminators)):
            logger.debug("Appending a period to help finalize segmentation of: '%s...'", processed_text[:50])
            processed_text += "." 
            
        all_sentences = self.text_segmenter.add_text(processed_text)
        
        # Handle any remaining text in TextSegmenter's buffer, considering it a final sentence/fragment.
        remainder = self.text_segmenter.get_remaining_text() # .strip() already in get_remaining_text
        if remainder: # Check if remainder is not just whitespace
            logger.debug(f"Appending remainder from TextSegmenter: '{remainder[:50]}...'")
            all_sentences.append(remainder)
        
        # Filter out any empty strings that might have resulted from segmentation.
        all_sentences = [s for s in all_sentences if s.strip()]

        if not all_sentences:
            logger.warning(f"No sentences were extracted from the text: '{full_text[:100]}...'")
            return

        logger.debug(f"Segmented into {len(all_sentences)} sentences: {all_sentences}")

        current_sentence_group: List[str] = []
        for i, sentence in enumerate(all_sentences):
            current_sentence_group.append(sentence)
            if len(current_sentence_group) >= self.sentences_per_chunk:
                chunk_to_yield = " ".join(current_sentence_group).strip()
                logger.debug(f"Yielding TTS chunk {i // self.sentences_per_chunk + 1}: '{chunk_to_yield[:100]}...'")
                yield chunk_to_yield
                current_sentence_group = []
        
        # Yield any remaining sentences as the last chunk
        if current_sentence_group:
            final_chunk_to_yield = " ".join(current_sentence_group).strip()
            logger.debug(f"Yielding final TTS chunk: '{final_chunk_to_yield[:100]}...'")
            yield final_chunk_to_yield