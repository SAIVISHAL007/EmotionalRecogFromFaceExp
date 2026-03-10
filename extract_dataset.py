import zipfile
import os

print('🔄 Extracting fer2013.zip...')
zip_path = r'c:\EmotionalRecogFromFaceExp\fer2013.zip'
extract_path = r'c:\EmotionalRecogFromFaceExp\data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print('✅ Extraction complete!')

# Verify structure
train_emotions = os.listdir(os.path.join(extract_path, 'train'))
test_emotions = os.listdir(os.path.join(extract_path, 'test'))

print(f'\n📊 Train emotions: {len([x for x in train_emotions if os.path.isdir(os.path.join(extract_path, "train", x))])} folders')
print(f'📊 Test emotions: {len([x for x in test_emotions if os.path.isdir(os.path.join(extract_path, "test", x))])} folders')
print(f'✅ Dataset ready for training!')
