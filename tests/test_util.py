"""
Tests for util.py module
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from aind_mri_targeting.util import check_output_path, err_if_files_exist, install_notebooks


class TestInstallNotebooks:
    """Test cases for install_notebooks function"""

    def test_install_notebooks_default_output(self, tmp_path, monkeypatch):
        """Test install_notebooks with default output (cwd)"""
        # Create a temporary notebooks directory structure
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()

        # Create test notebook files
        test_file1 = notebooks_dir / "test_notebook1.py"
        test_file2 = notebooks_dir / "test_notebook2.py"
        test_file1.write_text("# Test notebook 1")
        test_file2.write_text("# Test notebook 2")

        # Create a temporary working directory
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        # Change to the working directory
        monkeypatch.chdir(work_dir)

        # Mock the Path resolution to point to our test notebooks
        with patch("aind_mri_targeting.util.Path") as mock_path:
            notebooks_output = work_dir / "notebooks"
            mock_path.return_value.__truediv__.return_value = notebooks_output
            mock_resolve = mock_path.return_value.resolve.return_value
            mock_resolve.parent.parent.parent = tmp_path

            # Mock glob to return our test files
            with patch("aind_mri_targeting.util.glob.glob") as mock_glob:
                mock_glob.return_value = [str(test_file1), str(test_file2)]

                install_notebooks()

                # Check that the notebooks directory was created
                output_notebooks_dir = work_dir / "notebooks"
                assert output_notebooks_dir.exists()

                # Check that files were copied
                assert (output_notebooks_dir / "test_notebook1.py").exists()
                assert (output_notebooks_dir / "test_notebook2.py").exists()

    def test_install_notebooks_custom_output(self, tmp_path):
        """Test install_notebooks with custom output directory"""
        # Create a temporary notebooks directory structure
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()

        # Create test notebook files
        test_file = notebooks_dir / "test_notebook.py"
        test_file.write_text("# Test notebook")

        # Create custom output directory
        output_dir = tmp_path / "custom_output"
        output_dir.mkdir()

        # Mock the Path resolution and glob
        with patch("aind_mri_targeting.util.Path") as mock_path:
            notebooks_output = output_dir / "notebooks"
            mock_path.return_value.__truediv__.return_value = notebooks_output
            mock_resolve = mock_path.return_value.resolve.return_value
            mock_resolve.parent.parent.parent = tmp_path

            with patch("aind_mri_targeting.util.glob.glob") as mock_glob:
                mock_glob.return_value = [str(test_file)]

                install_notebooks(output=str(output_dir))

                # Check notebooks directory was created in custom location
                output_notebooks_dir = output_dir / "notebooks"
                assert output_notebooks_dir.exists()
                assert (output_notebooks_dir / "test_notebook.py").exists()

    def test_install_notebooks_skip_existing_files(self, tmp_path):
        """Test that existing files are skipped when force=False"""
        # Create notebooks directory and test file
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        test_file = notebooks_dir / "existing_notebook.py"
        test_file.write_text("# Original content")

        # Create output directory with existing file
        output_dir = tmp_path / "output"
        output_notebooks_dir = output_dir / "notebooks"
        output_notebooks_dir.mkdir(parents=True)
        existing_output_file = output_notebooks_dir / "existing_notebook.py"
        existing_output_file.write_text("# Existing content")

        # Mock Path and glob
        with patch("aind_mri_targeting.util.Path") as mock_path:
            mock_path.return_value.__truediv__.return_value = output_notebooks_dir
            mock_resolve = mock_path.return_value.resolve.return_value
            mock_resolve.parent.parent.parent = tmp_path
            mock_path.return_value.name = "existing_notebook.py"

            with patch("aind_mri_targeting.util.glob.glob") as mock_glob:
                mock_glob.return_value = [str(test_file)]

                with patch("builtins.print") as mock_print:
                    install_notebooks(output=str(output_dir), force=False)

                    # Should print skip message
                    expected_msg = "Skipping existing_notebook.py"
                    mock_print.assert_called_with(expected_msg)

                    # File content should remain unchanged
                    expected_content = "# Existing content"
                    assert existing_output_file.read_text() == expected_content

    def test_install_notebooks_force_overwrite(self, tmp_path):
        """Test that existing files are overwritten when force=True"""
        # Create notebooks directory and test file
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        test_file = notebooks_dir / "notebook.py"
        test_file.write_text("# New content")

        # Create output directory with existing file
        output_dir = tmp_path / "output"
        output_notebooks_dir = output_dir / "notebooks"
        output_notebooks_dir.mkdir(parents=True)
        existing_output_file = output_notebooks_dir / "notebook.py"
        existing_output_file.write_text("# Old content")

        # Mock shutil.copy2 to simulate copying
        with patch("aind_mri_targeting.util.shutil.copy2") as mock_copy:
            with patch("aind_mri_targeting.util.Path") as mock_path:
                mock_path.return_value.__truediv__.return_value = output_notebooks_dir
                mock_resolve = mock_path.return_value.resolve.return_value
                mock_resolve.parent.parent.parent = tmp_path
                mock_path.return_value.name = "notebook.py"

                with patch("aind_mri_targeting.util.glob.glob") as mock_glob:
                    mock_glob.return_value = [str(test_file)]

                    with patch("builtins.print") as mock_print:
                        install_notebooks(output=str(output_dir), force=True)

                        # Should print copy message
                        mock_print.assert_called_with("Copying notebook.py")

                        # Should call copy2
                        mock_copy.assert_called_once()

    def test_install_notebooks_creates_output_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist"""
        # Create notebooks directory
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        test_file = notebooks_dir / "test.py"
        test_file.write_text("# Test")

        # Output directory that doesn't exist yet
        output_dir = tmp_path / "new_output"

        with patch("aind_mri_targeting.util.Path") as mock_path:
            mock_path.return_value.__truediv__.return_value = output_dir / "notebooks"
            mock_resolve = mock_path.return_value.resolve.return_value
            mock_resolve.parent.parent.parent = tmp_path

            with patch("aind_mri_targeting.util.glob.glob") as mock_glob:
                mock_glob.return_value = [str(test_file)]

                install_notebooks(output=str(output_dir))

                # Directory should be created
                assert (output_dir / "notebooks").exists()


class TestCheckOutputPath:
    """Test cases for check_output_path function"""

    def test_check_output_path_valid_directory(self, tmp_path):
        """Test check_output_path with a valid directory"""
        result = check_output_path(str(tmp_path))
        assert isinstance(result, Path)
        assert result == Path(tmp_path)

    def test_check_output_path_none_uses_cwd(self):
        """Test check_output_path with None uses current working directory"""
        with patch("aind_mri_targeting.util.os.getcwd") as mock_getcwd:
            mock_getcwd.return_value = "/fake/cwd"
            with patch("aind_mri_targeting.util.os.path.isdir") as mock_isdir:
                mock_isdir.return_value = True

                result = check_output_path(None)

                assert isinstance(result, Path)
                assert str(result) == "/fake/cwd"
                mock_getcwd.assert_called_once()

    def test_check_output_path_invalid_directory(self, tmp_path):
        """Test check_output_path raises NotADirectoryError for invalid path"""
        invalid_path = tmp_path / "nonexistent"

        with pytest.raises(NotADirectoryError) as exc_info:
            check_output_path(str(invalid_path))

        assert "is not a directory" in str(exc_info.value)
        assert str(invalid_path) in str(exc_info.value)

    def test_check_output_path_file_not_directory(self, tmp_path):
        """Test check_output_path raises error when path is a file"""
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("test content")

        with pytest.raises(NotADirectoryError) as exc_info:
            check_output_path(str(file_path))

        assert "is not a directory" in str(exc_info.value)


class TestErrIfFilesExist:
    """Test cases for err_if_files_exist function"""

    def test_err_if_files_exist_no_files(self):
        """Test err_if_files_exist with non-existent files"""
        non_existent_files = ["/fake/path1.txt", "/fake/path2.txt"]

        # Should not raise any exception
        err_if_files_exist(non_existent_files)

    def test_err_if_files_exist_empty_list(self):
        """Test err_if_files_exist with empty file list"""
        # Should not raise any exception
        err_if_files_exist([])

    def test_err_if_files_exist_file_exists(self, tmp_path):
        """Test err_if_files_exist raises FileExistsError when file exists"""
        existing_file = tmp_path / "existing_file.txt"
        existing_file.write_text("test content")

        with pytest.raises(FileExistsError) as exc_info:
            err_if_files_exist([str(existing_file)])

        assert "already exists" in str(exc_info.value)
        assert "Set force=True to overwrite" in str(exc_info.value)
        assert str(existing_file) in str(exc_info.value)

    def test_err_if_files_exist_multiple_files_one_exists(self, tmp_path):
        """Test err_if_files_exist with multiple files where one exists"""
        existing_file = tmp_path / "existing.txt"
        existing_file.write_text("content")
        non_existing_file = tmp_path / "non_existing.txt"

        files_to_check = [str(non_existing_file), str(existing_file)]

        with pytest.raises(FileExistsError) as exc_info:
            err_if_files_exist(files_to_check)

        assert str(existing_file) in str(exc_info.value)

    def test_err_if_files_exist_custom_error_message(self, tmp_path):
        """Test err_if_files_exist with custom error message"""
        existing_file = tmp_path / "test.txt"
        existing_file.write_text("content")

        custom_message = "Custom error: {} is already there!"

        with pytest.raises(FileExistsError) as exc_info:
            err_if_files_exist([str(existing_file)], custom_message)

        assert "Custom error:" in str(exc_info.value)
        assert "is already there!" in str(exc_info.value)
        assert str(existing_file) in str(exc_info.value)

    def test_err_if_files_exist_returns_none_on_success(self):
        """Test that err_if_files_exist returns None when no files exist"""
        result = err_if_files_exist(["/fake/file1.txt", "/fake/file2.txt"])
        assert result is None
