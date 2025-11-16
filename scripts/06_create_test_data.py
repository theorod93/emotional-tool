"""
CREATE TEST DATASET FOR LOCAL TESTING
======================================
Purpose: Generate synthetic test data if full dataset unavailable
Date: November 2025
"""

import pandas as pd
import random
from pathlib import Path

def create_test_dataset(n_samples=1000, output_file='data/test_sample.csv'):
    """
    Generate synthetic emotion dataset for testing

    Args:
        n_samples: Number of samples to generate
        output_file: Output CSV file path

    Creates realistic emotion-labeled texts
    """
    print(f"\nGenerating {n_samples} test samples...")

    # 8 emotions from NRC lexicon
    emotions = ['Anger', 'Anticipation', 'Disgust', 'Fear', 
                'Joy', 'Sadness', 'Surprise', 'Trust']

    # Text templates for each emotion
    templates = {
        'Anger': [
            "This makes me so angry and frustrated!",
            "I cannot believe how infuriating this situation is.",
            "This is absolutely outrageous and unacceptable!",
            "I am furious about what happened today.",
            "This aggressive behavior is making me very angry."
        ],
        'Anticipation': [
            "I am looking forward to what comes next.",
            "I cannot wait to see what happens tomorrow!",
            "I am excited about the upcoming opportunities.",
            "I anticipate great things in the future.",
            "I am eager to start this new adventure."
        ],
        'Disgust': [
            "This is absolutely disgusting and revolting.",
            "I find this situation deeply repulsive.",
            "This makes me feel sick to my stomach.",
            "I am appalled by this terrible situation.",
            "This disgusting behavior is unacceptable."
        ],
        'Fear': [
            "I am really scared about what might happen.",
            "This situation terrifies me deeply.",
            "I am afraid something bad will occur.",
            "This frightening news has me worried.",
            "I fear the worst may come to pass."
        ],
        'Joy': [
            "I am so happy and delighted about this!",
            "This brings me such wonderful joy!",
            "I am thrilled and excited beyond words!",
            "This makes me incredibly happy!",
            "I am overjoyed with these fantastic results!"
        ],
        'Sadness': [
            "I feel so sad and disappointed about this.",
            "This situation makes me deeply melancholy.",
            "I am heartbroken by what has happened.",
            "This unfortunate news fills me with sorrow.",
            "I feel depressed and down about everything."
        ],
        'Surprise': [
            "Wow, I did not see that coming!",
            "What an unexpected turn of events!",
            "I am shocked and amazed by this!",
            "This surprising revelation is incredible!",
            "I cannot believe this astonishing news!"
        ],
        'Trust': [
            "I have complete faith in this decision.",
            "I trust that everything will work out fine.",
            "I believe in the reliability of this process.",
            "I have confidence that this is the right path.",
            "I trust the integrity of this approach."
        ]
    }

    # Generate samples
    data = []
    for i in range(n_samples):
        emotion = random.choice(emotions)
        template = random.choice(templates[emotion])

        # Add some variation
        text = template + f" Case {i}."

        # Add some noise (random words)
        if random.random() < 0.3:
            noise_words = ["indeed", "certainly", "really", "absolutely", "definitely"]
            text = text + " " + random.choice(noise_words)

        emotion_idx = emotions.index(emotion)

        data.append({
            'text': text,
            'labels': f'[{emotion_idx}]',
            'labels_str': f"['{emotion}']",
            'labels_source': f"['{emotion.lower()}']",
            'source': 'SYNTHETIC'
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save
    df.to_csv(output_file, index=False)

    print(f"✓ Created {len(df)} samples")
    print(f"✓ Saved to {output_file}")
    print(f"\nEmotion distribution:")
    print(df['labels_str'].value_counts())

    # Show sample
    print(f"\nSample rows:")
    print(df.head(3))

    return df

if __name__ == '__main__':
    print("="*80)
    print("CREATING TEST DATASET")
    print("="*80)

    # Generate 1000 samples (adjust as needed)
    df = create_test_dataset(n_samples=1000)

    print(f"\n✓ Test dataset ready!")
    print(f"  Use this file: data/test_sample.csv")
    print(f"  Next step: Run 03_preprocess_local.py")
