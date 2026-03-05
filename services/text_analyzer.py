import logging
from typing import Dict
from config import settings

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        # Maps simple keywords to specific emotion categories with an associated weight boost
        self.emotion_keywords = {
            "angry": ["kill", "hate", "angry", "mad", "furious", "stupid", "idiot", "die", "worst", "terrible", "shut up"],
            "happy": ["happy", "sweet", "joy", "beautiful", "wonderful", "great", "awesome", "love", "excellent", "glad", "nice", "good"],
            "sad": ["sad", "depressed", "cry", "miserable", "hurt", "pain", "sorry", "miss", "lonely", "tragic"],
            "fear": ["scared", "fear", "terrified", "spooky", "creepy", "afraid", "panic", "horrible", "nightmare"],
            "disgust": ["gross", "disgusting", "ew", "nasty", "vile", "sick", "repulsive", "awful"],
            "surprise": ["wow", "omg", "shocked", "surprised", "unbelievable", "really", "whoa", "crazy", "suddenly"]
        }
        
        # How much to boost an emotion if a keyword is found
        self.match_weight = 0.6  # This will strongly pull the prediction toward the matched emotion

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze transcribed text and return probability boosts for matching emotions.
        Returns a dictionary mapping emotion names to boost values.
        """
        boosts = {emotion: 0.0 for emotion in settings.EMOTION_LABELS}
        
        if not text or not isinstance(text, str):
            return boosts
            
        text_lower = text.lower()
        words = set(text_lower.replace(".", "").replace(",", "").replace("!", "").replace("?", "").split())
        
        if not words:
            return boosts
            
        logger.info(f"Analyzing text for emotion keywords: '{text}'")
        matches_found = False
        
        for emotion, keywords in self.emotion_keywords.items():
            if emotion not in boosts:
                continue
                
            for keyword in keywords:
                # Check for exact word matches or if a longer keyword phrase is in the text
                if keyword in words or (len(keyword.split()) > 1 and keyword in text_lower):
                    boosts[emotion] += self.match_weight
                    matches_found = True
                    logger.info(f"Matched keyword '{keyword}' for emotion '{emotion}'")
                    
        return boosts
