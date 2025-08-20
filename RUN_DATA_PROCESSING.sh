# Back dataset processing
unzip dataset/back.zip -d dataset/back
python datasets_processing/annotation_format.py --input_file dataset/back/valid/_annotations.csv --output_file dataset/back/valid/_annotations2.csv
python datasets_processing/annotation_format.py --input_file dataset/back/train/_annotations.csv --output_file dataset/back/train/_annotations2.csv
python datasets_processing/annotation_format.py --input_file dataset/back/test/_annotations.csv --output_file dataset/back/test/_annotations2.csv

# Front dataset processing
unzip dataset/front.zip -d dataset/front
python datasets_processing/annotation_format.py --input_file dataset/front/valid/_annotations.csv --output_file dataset/front/valid/_annotations2.csv
python datasets_processing/annotation_format.py --input_file dataset/front/train/_annotations.csv --output_file dataset/front/train/_annotations2.csv
python datasets_processing/annotation_format.py --input_file dataset/front/test/_annotations.csv --output_file dataset/front/test/_annotations2.csv