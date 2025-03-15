import os
import zipfile
import logging

from django.core.files.base import ContentFile
from plagiarism_model.services.testing import PlagiarismDetector

logger = logging.getLogger(__name__)

text_extractor = PlagiarismDetector()


def extract_text_from_file(file_instance):
    """
    Extract text from a file using the PlagiarismDetector's extraction method
    """
    try:
        file_path = file_instance.file.path

        # Use the extract_text_from_file method from PlagiarismDetector
        return text_extractor.extract_text_from_file(file_path)
    except ValueError:
        # If the file format is unsupported, log and return empty
        logger.info(f"Skipping unsupported file format: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        return ""


def extract_text_from_zip(zip_instance):
    try:
        text_data = []
        with zipfile.ZipFile(zip_instance.file.path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                try:
                    # Extract the file temporarily
                    temp_dir = os.path.dirname(zip_instance.file.path)
                    temp_file_path = os.path.join(temp_dir, os.path.basename(file_name))

                    with zip_ref.open(file_name) as source, open(temp_file_path, 'wb') as target:
                        target.write(source.read())

                    # Use the extract_text_from_file function
                    file_content = extract_text_from_file_path(temp_file_path)
                    if file_content:  # Only add if we got content
                        text_data.append(file_content)

                    # Clean up the temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

                except Exception as e:
                    logger.warning(f"Skipping file {file_name} in zip: {str(e)}")
                    continue

        return "\n".join(text_data)
    except Exception as e:
        logger.error(f"Error extracting text from ZIP: {str(e)}")
        return ""


def extract_text_from_file_path(file_path):
    """Helper function to extract text from a file path"""
    try:
        return text_extractor.extract_text_from_file(file_path)
    except ValueError:
        # If the file format is unsupported, log and skip
        logger.info(f"Skipping unsupported file format: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        return ""


def determine_file_type(extension):
    if extension == '.pdf':
        return 'pdf'
    elif extension == '.txt':
        return 'txt'
    elif extension == '.docx':
        return 'doc'
    elif extension == '.ipynb':
        return 'ipynb'
    else:
        return 'txt'  # Default fallback


def process_zip_upload(request, zip_serializer):
    """
    Helper function to process zip file uploads and create file records

    Args:
        request: The HTTP request object
        zip_serializer: The validated serializer with zip data

    Returns:
        tuple: (success_status (bool), data (dict), status_code (int))
    """
    from file_handling.models import UploadedFile
    try:
        zip_folder = zip_serializer.save(user=request.user)
        files_to_create = []

        # Process the zip file
        zip_path = zip_folder.file.path
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    # Skip directories and hidden files
                    if file_name.endswith('/') or file_name.startswith('__MACOSX') or file_name.startswith('.'):
                        continue

                    try:
                        # Extract file temporarily to analyze it
                        temp_dir = os.path.dirname(zip_path)
                        temp_file_path = os.path.join(temp_dir, os.path.basename(file_name))

                        with zip_ref.open(file_name) as source, open(temp_file_path, 'wb') as target:
                            target.write(source.read())

                        # Determine file type based on extension
                        file_extension = os.path.splitext(file_name)[1].lower()
                        file_type = determine_file_type(file_extension)

                        # Try to extract text to validate the file is supported
                        try:
                            content = extract_text_from_file_path(temp_file_path)
                            if content:  # File is valid and has content
                                # Create a Django file object from the temporary file
                                with open(temp_file_path, 'rb') as f:
                                    django_file = f.read()

                                # Create the file record with the filename from the zip
                                file_obj = UploadedFile(
                                    user=request.user,
                                    file_type=file_type,
                                    zip_folder=zip_folder
                                )

                                # Save file with proper name from zip archive
                                file_obj.file.save(os.path.basename(file_name), ContentFile(django_file))
                                files_to_create.append(file_obj)
                        except ValueError:
                            logger.info(f"Skipping unsupported file format: {file_name}")

                        # Clean up temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

                    except Exception as e:
                        logger.warning(f"Error processing file {file_name} from zip: {str(e)}")
                        continue

        # Process files in the background
        from file_handling.tasks import process_file
        for uploaded_file in files_to_create:
            process_file.delay(uploaded_file.guid)

        return True, {
            "message": f"ZIP uploaded and processing started. Found {len(files_to_create)} valid files.",
            "data": zip_serializer.data,
        }, 201

    except Exception as e:
        logger.error(f"Error processing zip upload: {str(e)}")
        return False, {
            "message": "Error processing zip upload.",
            "error": str(e),
        }, 500
