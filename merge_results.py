#!/usr/bin/env python3
"""
Merge all best_metrics CSV files into a single Excel file.

Usage:
    python merge_results.py
    python merge_results.py --input_dir ./outputs --output_file all_experiments.xlsx
"""

import pandas as pd
import glob
import argparse
import os


def merge_best_metrics(input_dir='./outputs', output_file='all_experiments.xlsx'):
    """
    Merge all best_metrics CSV files into a single Excel file.
    
    Args:
        input_dir: Directory containing CSV files
        output_file: Output Excel filename
    """
    # Find all best_metrics CSV files
    pattern = os.path.join(input_dir, 'best_metrics_*.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"❌ No best_metrics CSV files found in {input_dir}")
        return None
    
    print(f"📂 Found {len(csv_files)} experiment result(s):")
    for f in csv_files:
        print(f"   • {os.path.basename(f)}")
    
    # Read and combine all CSVs
    all_results = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add source filename for reference (optional)
            df['Source File'] = os.path.basename(csv_file)
            all_results.append(df)
        except Exception as e:
            print(f"   ⚠️ Error reading {csv_file}: {e}")
    
    if not all_results:
        print("❌ No valid CSV files could be read")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns to put experiment config first
    config_cols = ['Model', 'Batch Size', 'Learning Rate', 'Epochs', 
                   'Model Size (MB)', 'Total Parameters', 'Trainable Parameters']
    result_cols = ['Best Fold', 'Best Epoch']
    val_cols = [col for col in combined_df.columns if col.startswith('Val')]
    train_cols = [col for col in combined_df.columns if col.startswith('Train')]
    other_cols = ['Source File']
    
    # Build ordered column list (only include columns that exist)
    ordered_cols = []
    for col in config_cols + result_cols + val_cols + train_cols + other_cols:
        if col in combined_df.columns:
            ordered_cols.append(col)
    
    # Add any remaining columns not in our ordered list
    for col in combined_df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    combined_df = combined_df[ordered_cols]
    
    # Save to Excel
    output_path = os.path.join(input_dir, output_file) if not os.path.dirname(output_file) else output_file
    
    # Use openpyxl for Excel formatting
    try:
        from openpyxl.utils import get_column_letter
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='All Experiments', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['All Experiments']
            for idx, col in enumerate(combined_df.columns):
                # Calculate max length, handling NaN values
                col_lengths = combined_df[col].astype(str).map(len)
                col_max_len = col_lengths.max()
                
                # Handle NaN or empty columns
                if pd.isna(col_max_len):
                    col_max_len = 0
                
                max_length = max(int(col_max_len), len(str(col))) + 2
                
                # Get proper column letter (works for any number of columns)
                column_letter = get_column_letter(idx + 1)
                worksheet.column_dimensions[column_letter].width = min(max_length, 50)
        
        print(f"\n✅ Merged results saved to: {output_path}")
        
    except ImportError:
        # Fallback to CSV if openpyxl not available
        output_path = output_path.replace('.xlsx', '.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Merged results saved to: {output_path}")
        print("   (Saved as CSV because openpyxl is not installed)")
        print("   Install with: pip install openpyxl")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Total experiments: {len(combined_df)}")
    
    if 'Val F1 (weighted)' in combined_df.columns:
        best_idx = combined_df['Val F1 (weighted)'].idxmax()
        best_row = combined_df.loc[best_idx]
        print(f"\n🏆 Best Experiment:")
        print(f"   Model: {best_row.get('Model', 'N/A')}")
        print(f"   Batch Size: {best_row.get('Batch Size', 'N/A')}")
        print(f"   Learning Rate: {best_row.get('Learning Rate', 'N/A')}")
        print(f"   Val F1 (weighted): {best_row.get('Val F1 (weighted)', 'N/A'):.4f}")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge all best_metrics CSV files into a single Excel file"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./outputs',
        help='Directory containing CSV files (default: ./outputs)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='all_experiments.xlsx',
        help='Output Excel filename (default: all_experiments.xlsx)'
    )
    
    args = parser.parse_args()
    merge_best_metrics(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()