import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from model.predict import validate_grading_inputs

class GradingInputTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(patient_id='p1',case_id='1',slice_id='74',slice_order=74,image_path='1/img-00003-00074.png')])

    def check(self, frame, file_exists=lambda p: True):
        with patch.object(Path,'is_dir',return_value=True), patch.object(Path,'is_file',file_exists), patch('model.predict.pd.read_csv',return_value=frame):
            return validate_grading_inputs('manifest.csv',Path('roi'),Path('masks'))

    def test_valid_nested_bilateral_pairs(self):
        self.assertEqual(len(self.check(self.frame())),1)

    def test_missing_right_mask_is_not_skipped(self):
        with self.assertRaisesRegex(FileNotFoundError,'right_mask'):
            self.check(self.frame(),lambda p: not str(p).endswith('_right_mask.png'))

    def test_empty_manifest(self):
        with self.assertRaisesRegex(ValueError,'empty'):
            self.check(self.frame().iloc[:0])

    def test_missing_directory(self):
        with patch.object(Path,'is_dir',return_value=False):
            with self.assertRaisesRegex(FileNotFoundError,'ROI directory'):
                validate_grading_inputs('manifest.csv',Path('absent'),Path('masks'))

    def test_conflicting_patient(self):
        frame=pd.concat([self.frame(),self.frame()],ignore_index=True)
        frame.loc[1,['patient_id','slice_id','slice_order','image_path']]=['p2','75',75,'1/img-00003-00075.png']
        with self.assertRaisesRegex(ValueError,'exactly one'):
            self.check(frame)

    def test_nonfinite_order(self):
        frame=self.frame(); frame.loc[0,'slice_order']=float('inf')
        with self.assertRaisesRegex(ValueError,'integers'):
            self.check(frame)
