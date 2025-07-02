# text_segmenter.py
import pysbd  # Ensure 'pip install pysbd'
import logging

logger = logging.getLogger(__name__)

class TextSegmenter:
    """
    Identifies complete sentences from text.
    This version is simplified for direct use by SentenceChunkerForTTS.
    """
    def __init__(self, language="en"):
        try:
            self.segmenter = pysbd.Segmenter(language=language, clean=False, char_span=False)
        except Exception as e:
            logger.error(f"Failed to initialize pysbd.Segmenter for language '{language}'. Is 'pysbd' and its language models installed? Error: {e}")
            # Fallback or re-raise, depending on desired strictness. For now, let it proceed and fail later if segmenter is None.
            # Or, more robustly: raise RuntimeError(f"pysbd.Segmenter init failed: {e}") from e
            self.segmenter = None # Or handle this more gracefully
        self.buffer = ""
        self.terminators = {'.', '?', '!'} # Used by SentenceChunkerForTTS

    def add_text(self, text_chunk: str) -> list[str]:
        """
        Processes a text chunk and returns a list of identified sentences.
        For use with SentenceChunkerForTTS, this is often called with the full text block.
        """
        if self.segmenter is None:
            logger.error("pysbd.Segmenter not initialized in TextSegmenter. Cannot segment text.")
            return [text_chunk] # Fallback: return the whole chunk as one "sentence"

        if not isinstance(text_chunk, str):
            return []
        
        self.buffer += text_chunk # Accumulate

        if not self.buffer.strip():
            return []

        # Segment the entire current buffer
        potential_sentences = self.segmenter.segment(self.buffer)

        if not potential_sentences:
            # It's possible the buffer only contains whitespace or pysbd couldn't segment.
            # If the buffer was non-empty but only whitespace, it should be cleared.
            if not self.buffer.strip():
                 self.buffer = ""
            return []

        # Heuristic: If the original input `text_chunk` (which might be the full text in this context)
        # ends with a terminator, or if pysbd broke it into multiple segments,
        # assume all segments up to the last one are complete.
        # The last segment is complete if the original text chunk itself appeared to end completely.
        
        last_original_char = self.buffer.strip()[-1] if self.buffer.strip() else ''

        if last_original_char in self.terminators or len(potential_sentences) > 1:
            # If the whole buffer seems to end conclusively, or we got multiple sentences,
            # assume all pysbd segments are what we want.
            self.buffer = "" # Clear buffer as it's all processed into these sentences
            return [s.strip() for s in potential_sentences if s.strip()]
        else:
            # Single segment returned, and original text didn't clearly end.
            # This segment might be incomplete. Keep it in buffer.
            # However, for chunk_text, it often passes text that *should* be fully processed.
            # If only one sentence is returned by pysbd for the entire input,
            # consider it a single sentence.
            if len(potential_sentences) == 1:
                self.buffer = "" # Assume the single sentence is complete for this context
                return [s.strip() for s in potential_sentences if s.strip()]
            else: # Should not be reached if len(potential_sentences) > 1 is handled above
                self.buffer = "" 
                return [s.strip() for s in potential_sentences if s.strip()]


    def get_remaining_text(self) -> str:
        """
        Returns any text remaining in the internal buffer.
        """
        return self.buffer.strip()

    def clear_buffer(self):
        """
        Clears the internal text buffer.
        """
        self.buffer = ""