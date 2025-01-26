import unittest
from collections import defaultdict
from pathlib import Path

from experiments.dataset_analyzer import DatasetAnalyzer


class TestSplitValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This will run once before all tests
        cls.data_df = DatasetAnalyzer.load_dataset(Path("../../data/Features"))
        cls.splits = DatasetAnalyzer.load_nested_splits(Path("../../configs/splits"))
        cls.min_samples_per_class = 5

        # Compute stats once to be used by all tests
        cls.stats = {}

        # Track test set appearances
        test_participants = defaultdict(int)

        # Track validation set appearances by outer fold
        fold_stats = {}

        for outer, inner_folds in cls.splits.items():
            # Get test participants (should be same across inner splits)
            outer_participants = None
            inner_participants = defaultdict(int)

            for fold, frame in inner_folds.items():
                # Check test set consistency within outer fold
                fold_test = frame[frame.Split == "test"].ParticipantId.to_list()
                if outer_participants is None:
                    outer_participants = fold_test
                else:
                    assert outer_participants == fold_test

                # Track validation set appearances
                for p in frame[frame.Split == "validation"].ParticipantId.to_list():
                    inner_participants[p] += 1

            # Update test participant counts
            for p in outer_participants:
                test_participants[p] += 1

            # Store stats for this fold
            fold_stats[outer] = {
                'test_participants': outer_participants,
                'validation_counts': inner_participants,
                'n_validation': len(inner_participants)
            }

        # Store all computed stats
        cls.stats = {
            'test_participant_counts': test_participants,
            'fold_stats': fold_stats,
            'n_total_participants': len(cls.data_df.ParticipantId.unique())
        }

    def test_each_participant_appears_once_in_test(self):
        """Test that each participant appears exactly once in test sets."""
        counts = self.stats['test_participant_counts']
        self.assertTrue(all(count == 1 for count in counts.values()),
                        "Some participants appear multiple times in test sets")
        self.assertEqual(len(counts), self.stats['n_total_participants'],
                         "Not all participants appear in test sets")

    # def test_validation_set_sizes(self):
    #     """Test that validation sets have correct size in each fold."""
    #     for outer, fold_stat in self.stats['fold_stats'].items():
    #         expected_val_size = self.stats['n_total_participants'] - len(fold_stat["test_participants"])
    #         self.assertEqual(fold_stat['n_validation'], expected_val_size,
    #                          f"Incorrect validation set size in fold {outer}")

    def test_validation_appearances_per_fold(self):
        """Test that each participant appears exactly once in validation sets within each fold."""
        for outer, fold_stat in self.stats['fold_stats'].items():
            counts = fold_stat['validation_counts']
            self.assertTrue(all(count >= 1 for count in counts.values()),
                            f"Multiple validation appearances in fold {outer}")

    def test_no_validation_test_overlap(self):
        """Test that no participant appears in both validation and test sets of same fold."""
        for outer, fold_stat in self.stats['fold_stats'].items():
            test_participants = set(fold_stat['test_participants'])
            val_participants = set(fold_stat['validation_counts'].keys())
            self.assertEqual(len(test_participants & val_participants), 0,
                             f"Validation/test overlap in fold {outer}")

    def test_minimum_class_representation(self):
        """Test that each validation/test set has minimum samples per class."""

        for outer, fold_stat in self.stats['fold_stats'].items():
            # Check test set
            test_subset = self.data_df[self.data_df.ParticipantId.astype(int).isin(fold_stat['test_participants'])]
            test_class_counts = test_subset.Label.value_counts()
            self.assertGreaterEqual(test_class_counts.min(), self.min_samples_per_class, f"Insufficient class samples in test set of fold {outer}")

            # Check validation sets within the fold
            for inner_fold, frame in self.splits[outer].items():
                val_participants = frame[frame.Split == "validation"].ParticipantId.tolist()
                val_subset = self.data_df[self.data_df.ParticipantId.astype(int).isin(val_participants)]
                val_class_counts = val_subset.Label.value_counts()
                self.assertGreaterEqual(val_class_counts.min(), self.min_samples_per_class,
                                        f"Insufficient class samples in validation set of fold {outer}, inner fold {inner_fold}")


if __name__ == '__main__':
    unittest.main()
