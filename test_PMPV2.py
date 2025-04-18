import unittest
import json
import PMPV2 as pmpv
from unittest.mock import patch, mock_open


class TestLoadData(unittest.TestCase):

    def test_valid_json(self):
        mock_json = '{"key": "value"}'
        with patch("builtins.open", mock_open(read_data=mock_json)):
            data, error = pmpv.load_json("dummy.json")
            self.assertEqual(data, {"key": "value"})
            self.assertIsNone(error)

    def test_file_not_found(self):
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            data, error = pmpv.load_json("missing.json")
            self.assertIsNone(data)
            self.assertIn("File not found", error)

    def test_invalid_json(self):
        mock_invalid_json = "{key: value}"
        with patch("builtins.open", mock_open(read_data=mock_invalid_json)):
            with patch("json.load", side_effect=json.JSONDecodeError("Invalid JSON", "", 0)):
                data, error = pmpv.load_json("invalid.json")
                self.assertIsNone(data)
                self.assertIn("Invalid JSON", error)

    def test_permission_error(self):
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            data, error = pmpv.load_json("protected.json")
            self.assertIsNone(data)
            self.assertIn("Permission denied", error)


if __name__ == "__main__":
    unittest.main()
