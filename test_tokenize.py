
from utils import text

if __name__ == '__main__':
    # Test the text_to_sequence and sequence_to_text functions
    test_text = "Hello, world! {HH AH0 L OW1} This is a test."
    cleaner_names = ['english_cleaners']
    
    sequence = text.text_to_sequence(test_text, cleaner_names)
    print("Sequence:", sequence)

    reconstructed_text = text.sequence_to_text(sequence)
    print("Reconstructed Text:", reconstructed_text)
