import pandas as pd 
import argparse



def transform_coordinates(input_file, output_file):
    """
    Transform pixel coordinates to normalized coordinates (0-1 range).
    
    Args:
    input_file (str): Path to the input CSV file with pixel coordinates
    output_file (str): Path to save the output CSV file with normalized coordinates
    """
    # Read the input file
    df = pd.read_csv(input_file)
    transformed_data = []
    
    for filename, group in df.groupby('filename'):
        # Get image width and height
        width = group['width'].iloc[0]
        height = group['height'].iloc[0]
        
        for _, row in group.iterrows():
            # Transform pixel coordinates to normalized coordinates (0-1 range)
            xmin_norm = row['xmin'] / width
            xmax_norm = row['xmax'] / width
            ymin_norm = row['ymin'] / height
            ymax_norm = row['ymax'] / height
            
            transformed_row = {
                'ImageID': filename,  
                'ClassName': "people",  
                'Confidence': 1,  
                'XMin': xmin_norm,
                'XMax': xmax_norm,
                'YMin': ymin_norm,
                'YMax': ymax_norm
            }
            
            transformed_data.append(transformed_row)
    
    transformed_df = pd.DataFrame(transformed_data)
    column_order = ['ImageID', 'ClassName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax']
    transformed_df = transformed_df[column_order]
    
    # Save to CSV
    transformed_df.to_csv(output_file, index=False, sep='\t')
    
    print(f"Transformation complete. Output saved to {output_file}")
    print(transformed_df)

def main():
    parser = argparse.ArgumentParser(description="Transform pixel coordinates to normalized coordinates.")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input CSV file with pixel coordinates."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to save the output CSV file with normalized coordinates."
    )

    args = parser.parse_args()

    transform_coordinates(args.input_file, args.output_file)

if __name__ == "__main__":
    main()