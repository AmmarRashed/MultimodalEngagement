from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DatasetAnalyzer:
    def __init__(self, dataset_root: str, splits_root: str):
        self.dataset_root = Path(dataset_root)
        self.splits_root = Path(splits_root)
        self.data_df = self.load_dataset(self.dataset_root)
        self.splits = self.load_nested_splits(self.splits_root)

    @staticmethod
    def load_dataset(root_dir) -> pd.DataFrame:
        """Load all dataset information into a DataFrame."""
        data = []
        for participant_dir in root_dir.glob("*"):
            if not participant_dir.is_dir():
                continue

            participant_id = participant_dir.name

            for video_file in participant_dir.glob("*.npz"):
                filename_parts = video_file.stem.split('_')
                label = int(filename_parts[2])

                # if label != 2:  # Ignore neutral class
                binary_label = 1 if label > 2 else 0
                data.append({
                    'ParticipantId': participant_id,  # Keep as string
                    'VideoId': filename_parts[1],
                    'Label': binary_label
                })

        return pd.DataFrame(data)

    @staticmethod
    def load_nested_splits(root_dir) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load nested split structure into a dictionary."""
        splits = {}
        for fold_dir in root_dir.glob("fold_*"):
            fold_idx = fold_dir.name.split('_')[1]
            splits[fold_idx] = {}

            for split_file in fold_dir.glob("*.csv"):
                inner_fold_idx = split_file.stem
                splits[fold_idx][inner_fold_idx] = pd.read_csv(split_file)

        return splits

    def overall_statistics(self) -> pd.DataFrame:
        """Calculate overall dataset statistics."""
        stats = {
            'Total Participants': len(self.data_df['ParticipantId'].unique()),
            'Total Videos': len(self.data_df),
            'Class 0 Videos': sum(self.data_df['Label'] == 0),
            'Class 1 Videos': sum(self.data_df['Label'] == 1),
            'Class Distribution (% Class 1)': (sum(self.data_df['Label'] == 1) / len(self.data_df)) * 100
        }
        return pd.DataFrame([stats])

    def participant_statistics(self) -> pd.DataFrame:
        """Calculate participant-level statistics."""
        participant_stats = []

        for participant_id in self.data_df['ParticipantId'].unique():
            participant_data = self.data_df[self.data_df['ParticipantId'] == participant_id]

            stats = {
                'ParticipantId': participant_id,
                'Total Videos': len(participant_data),
                'Class 0 Videos': sum(participant_data['Label'] == 0),
                'Class 1 Videos': sum(participant_data['Label'] == 1),
                'Class Distribution (% Class 1)': (sum(participant_data['Label'] == 1) / len(participant_data)) * 100
            }
            participant_stats.append(stats)

        return pd.DataFrame(participant_stats)

    def visualize_overall_distribution(self) -> plt.Figure:
        """Create visualization of overall class distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))

        class_counts = self.data_df['Label'].value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)

        ax.set_title('Overall Class Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Videos')

        return fig

    def visualize_participant_distributions(self) -> plt.Figure:
        """Create visualization of per-participant class distributions."""
        participant_stats = self.participant_statistics()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Videos per participant
        sns.histplot(data=participant_stats, x='Total Videos', ax=ax1)
        ax1.set_title('Distribution of Videos per Participant')

        # Class distribution per participant
        sns.histplot(data=participant_stats, x='Class Distribution (% Class 1)', ax=ax2)
        ax2.set_title('Distribution of Class 1 Percentage per Participant')

        plt.tight_layout()
        return fig

    def analyze_fold_distributions(self) -> Dict[str, pd.DataFrame]:
        """Analyze class distributions for each outer fold across their inner folds."""
        fold_stats = {}

        for outer_fold_idx, inner_splits in self.splits.items():
            train_stats = []
            val_stats = []

            # Get test set participants (same for all inner folds)
            test_participants = inner_splits[list(inner_splits.keys())[0]][
                inner_splits[list(inner_splits.keys())[0]]['Split'] == 'test'
                ]['ParticipantId'].tolist()
            test_data = self.data_df[self.data_df['ParticipantId'].astype(int).isin(test_participants)]

            # Calculate test set statistics (constant for this outer fold)
            test_stats = {
                'total_samples': len(test_data),
                'class_0_samples': sum(test_data['Label'] == 0),
                'class_1_samples': sum(test_data['Label'] == 1),
                'class_1_ratio': sum(test_data['Label'] == 1) / len(test_data),
                'n_participants': len(test_participants)
            }

            # Analyze each inner fold
            for inner_fold_idx, split_df in inner_splits.items():
                # Get participants for each split
                train_participants = split_df[split_df['Split'] == 'train']['ParticipantId'].tolist()
                val_participants = split_df[split_df['Split'] == 'validation']['ParticipantId'].tolist()

                # Get corresponding data
                train_data = self.data_df[self.data_df['ParticipantId'].astype(int).isin(train_participants)]
                val_data = self.data_df[self.data_df['ParticipantId'].astype(int).isin(val_participants)]

                # Calculate statistics for train set
                train_stats.append({
                    'inner_fold': inner_fold_idx,
                    'total_samples': len(train_data),
                    'class_0_samples': sum(train_data['Label'] == 0),
                    'class_1_samples': sum(train_data['Label'] == 1),
                    'class_1_ratio': sum(train_data['Label'] == 1) / len(train_data),
                    'n_participants': len(train_participants)
                })

                # Calculate statistics for validation set
                val_stats.append({
                    'inner_fold': inner_fold_idx,
                    'total_samples': len(val_data),
                    'class_0_samples': sum(val_data['Label'] == 0),
                    'class_1_samples': sum(val_data['Label'] == 1),
                    'class_1_ratio': sum(val_data['Label'] == 1) / len(val_data),
                    'n_participants': len(val_participants)
                })

            # Convert to DataFrames
            train_df = pd.DataFrame(train_stats)
            val_df = pd.DataFrame(val_stats)

            # Calculate summary statistics
            summary_stats = ['min', 'max', 'mean', '50%', 'std']
            train_summary = train_df.describe().loc[summary_stats, :]
            val_summary = val_df.describe().loc[summary_stats, :]

            # Create test summary with same stats as train/val for consistency
            test_summary = pd.DataFrame({k: v for k, v in test_stats.items()}, index=['mean'])
            for stat in summary_stats:
                if stat != 'mean':
                    test_summary.loc[stat] = test_summary.loc['mean'] if stat in ['min', 'max', '50%'] else 0

            fold_stats[outer_fold_idx] = {
                'train': train_summary,
                'validation': val_summary,
                'test': test_summary,
                'train_raw': train_df,
                'val_raw': val_df
            }

        return fold_stats

    def visualize_fold_distributions(self, fold_stats: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, plt.Figure]:
        """Create visualizations for fold distributions."""
        figures = {}

        # Prepare data for plotting
        train_data = []
        val_data = []
        test_data = []

        for fold_idx, stats in sorted(fold_stats.items()):
            # Add fold index to raw data
            train_fold_data = stats['train_raw'].copy()
            train_fold_data['outer_fold'] = fold_idx
            train_data.append(train_fold_data)

            val_fold_data = stats['val_raw'].copy()
            val_fold_data['outer_fold'] = fold_idx
            val_data.append(val_fold_data)

            # Add test data (same for all inner folds)
            test_fold_data = pd.DataFrame([{
                'outer_fold': fold_idx,
                'class_1_ratio': stats['test'].loc['mean', 'class_1_ratio'],
                'total_samples': stats['test'].loc['mean', 'total_samples']
            }])
            test_data.append(test_fold_data)

        train_df = pd.concat(train_data)
        val_df = pd.concat(val_data)
        test_df = pd.concat(test_data)

        # Create boxplots for class distribution
        fig_class_dist, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))

        sns.boxplot(data=train_df, x='outer_fold', y='class_1_ratio', ax=ax1)
        ax1.set_title('Class 1 Ratio Distribution Across Inner Folds (Train)')
        ax1.set_ylabel('Class 1 Ratio')

        sns.boxplot(data=val_df, x='outer_fold', y='class_1_ratio', ax=ax2)
        ax2.set_title('Class 1 Ratio Distribution Across Inner Folds (Validation)')
        ax2.set_ylabel('Class 1 Ratio')

        sns.barplot(data=test_df, x='outer_fold', y='class_1_ratio', ax=ax3)
        ax3.set_title('Class 1 Ratio (Test)')
        ax3.set_ylabel('Class 1 Ratio')

        plt.tight_layout()
        figures['class_distribution'] = fig_class_dist

        # Create plots for sample counts
        fig_samples, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))

        sns.boxplot(data=train_df, x='outer_fold', y='total_samples', ax=ax1)
        ax1.set_title('Total Samples Distribution Across Inner Folds (Train)')
        ax1.set_ylabel('Number of Samples')

        sns.boxplot(data=val_df, x='outer_fold', y='total_samples', ax=ax2)
        ax2.set_title('Total Samples Distribution Across Inner Folds (Validation)')
        ax2.set_ylabel('Number of Samples')

        sns.barplot(data=test_df, x='outer_fold', y='total_samples', ax=ax3)
        ax3.set_title('Total Samples (Test)')
        ax3.set_ylabel('Number of Samples')

        plt.tight_layout()
        figures['sample_distribution'] = fig_samples

        return figures

    def get_fold_analysis_df(self, fold_stats: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """Create a DataFrame summarizing fold statistics."""
        analysis_data = []

        for fold_idx, stats in sorted(fold_stats.items()):
            for split_type in ['train', 'validation', 'test']:
                for stat_type in ['min', 'max', 'mean', '50%', 'std']:
                    row_data = {
                        'outer_fold': fold_idx,
                        'split_type': split_type,
                        'stat_type': stat_type
                    }

                    # Add all metrics
                    for col in stats[split_type].columns:
                        row_data[col] = stats[split_type].loc[stat_type, col]

                    analysis_data.append(row_data)

        return pd.DataFrame(analysis_data)

def plot_with_error_bars(data: pd.DataFrame, x_col: str, y_col: str,
                         title: str) -> plt.Figure:
    """Helper function to create plots with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    means = data.groupby(x_col)[y_col].mean()
    sems = data.groupby(x_col)[y_col].sem()

    sns.barplot(x=means.index, y=means.values, ax=ax)
    ax.errorbar(x=range(len(means)), y=means.values,
                yerr=sems.values * 1.96, fmt='none', color='black',
                capsize=5)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    return fig
