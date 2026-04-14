import os
if os.path.exists('best_audio_classifier.pth'):
    print("Model found!")
    print(f"Size: {os.path.getsize('best_audio_classifier.pth') / (1024*1024):.2f} MB")
else:
    print("Model not found in current directory")