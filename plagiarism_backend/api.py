from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser

from core.base_viewset import BaseViewSet
from core.utils import determine_file_type
from plagiarism_model.services.testing import PlagiarismDetector
from file_handling.models import UploadedFile, FileContent

import os
import logging
from drf_spectacular.utils import extend_schema

logger = logging.getLogger(__name__)


class PlagiarismViewSet(BaseViewSet):

    @extend_schema(
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'suspicious_text': {
                        'type': 'string',
                        'description': 'Text to check for plagiarism'
                    },
                    'source_text': {
                        'type': 'string',
                        'description': 'Original text to compare against'
                    }
                },
                'required': ['suspicious_text', 'source_text']
            }
        },
        responses={
            200: {
                'description': 'Plagiarism check completed successfully',
                'content': {
                    'application/json': {
                        'example': {
                            'message': 'Plagiarism check successful',
                            'plagiarism': 0.85
                        }
                    }
                }
            },
            400: {'description': 'Missing required text inputs'},
            500: {'description': 'Server error during plagiarism check'}
        },
        description='Compares two text inputs to detect plagiarism'
    )
    @action(detail=False, methods=['POST'], url_path='text-to-text', url_name='text-to-text')
    def text_to_text(self, request, *args, **kwargs):
        try:
            # Extract text inputs from request
            suspicious_text = request.data.get('suspicious_text')
            source_text = request.data.get('source_text')

            # Validate inputs
            if not suspicious_text or not source_text:
                return Response(
                    {
                        "error": "Both suspicious_text and source_text are required"
                    },
                    status=400
                )

            # Initialize detector
            detector = PlagiarismDetector()

            # Check for plagiarism
            result = detector.check_plagiarism(source_text, suspicious_text)

            return Response(
                {
                    "message": "Plagiarism check successful",
                    "plagiarism": result
                },
                status=200
            )

        except Exception as e:
            logger.error(f"Error in text-to-text plagiarism detection: {str(e)}")
            return Response(
                {
                    "error": f"Failed to process plagiarism check: {str(e)}"
                },
                status=500
            )

    @extend_schema(
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'suspicious_text': {
                        'type': 'string',
                        'description': 'Text to check for plagiarism'
                    },
                    'source_file': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'Original file to compare against (PDF, DOCX, TXT)'
                    }
                },
                'required': ['suspicious_text', 'source_file']
            }
        },
        responses={
            200: {
                'description': 'Plagiarism check completed successfully',
                'content': {
                    'application/json': {
                        'example': {
                            'message': 'Plagiarism check successful',
                            'plagiarism': {
                                'verdict': 'Plagiarism Detected',
                                'confidence': 0.85,
                                'is_plagiarized': True,
                                'similarity_score': 0.85
                            }
                        }
                    }
                }
            },
            400: {'description': 'Missing required inputs or invalid file format'},
            500: {'description': 'Server error during plagiarism check'}
        },
        description='Compares text input against a file for plagiarism detection'
    )
    @action(detail=False, methods=['POST'], url_path='text-to-file', url_name='text-to-file', parser_classes=[MultiPartParser, FormParser])
    def text_to_file(self, request, *args, **kwargs):
        try:
            # Extract inputs from request
            suspicious_text = request.data.get('suspicious_text')
            source_file = request.FILES.get('source_file')

            # Validate inputs
            if not suspicious_text:
                return Response(
                    {"error": "Suspicious text is required"},
                    status=400
                )

            if not source_file:
                return Response(
                    {"error": "Source file is required"},
                    status=400
                )

            # Validate file type
            file_extension = os.path.splitext(source_file.name)[1].lower()
            supported_formats = ['.pdf', '.docx', '.txt', '.ipynb']

            if file_extension not in supported_formats:
                return Response(
                    {"error": f"Unsupported file format. Please upload {', '.join(supported_formats)}"},
                    status=400
                )

            # Determine file type
            if file_extension == '.pdf':
                file_type = 'pdf'
            elif file_extension == '.txt':
                file_type = 'txt'
            elif file_extension == '.docx':
                file_type = 'doc'
            elif file_extension == '.ipynb':
                file_type = 'ipynb'
            else:
                file_type = 'txt'  # Default fallback

            # Check if a file with the same content already exists
            # First save the file temporarily to extract its content
            temp_file = UploadedFile.objects.create(
                file=source_file,
                file_type=file_type,
                user=request.user,
                processed=False
            )

            # Initialize detector
            detector = PlagiarismDetector()

            # Extract text from file
            source_text = None
            try:
                source_text = detector.extract_text_from_file(temp_file.file.path)

                # Check if this content already exists
                existing_content = FileContent.objects.filter(content=source_text).first()

                if existing_content:
                    # Use the existing file
                    uploaded_file = existing_content.uploaded_file
                    # Delete the temporary file since we'll use the existing one
                    temp_file.file.delete()
                    temp_file.delete()
                else:
                    # Keep the new file
                    uploaded_file = temp_file
                    # Save extracted content to FileContent model
                    FileContent.objects.create(
                        uploaded_file=uploaded_file,
                        content=source_text
                    )

                # Mark file as processed
                uploaded_file.processed = True
                uploaded_file.save()

                # Check for plagiarism
                result = detector.check_plagiarism(source_text, suspicious_text)

                return Response(
                    {
                        "message": "Plagiarism check successful",
                        "plagiarism": result
                    },
                    status=200
                )

            except Exception as e:
                temp_file.file.delete()
                temp_file.delete()
                logger.error(f"Error extracting text from file: {str(e)}")
                return Response(
                    {"error": f"Failed to extract text from file: {str(e)}"},
                    status=500
                )

        except Exception as e:
            logger.error(f"Error in text-to-file plagiarism detection: {str(e)}")
            return Response(
                {"error": f"Failed to process plagiarism check: {str(e)}"},
                status=500
            )

    @extend_schema(
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'suspicious_file': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'File to check for plagiarism (PDF, DOCX, TXT)'
                    },
                    'source_file': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'Original file to compare against (PDF, DOCX, TXT)'
                    }
                },
                'required': ['suspicious_file', 'source_file']
            }
        },
        responses={
            200: {
                'description': 'Plagiarism check completed successfully',
                'content': {
                    'application/json': {
                        'example': {
                            'message': 'Plagiarism check successful',
                            'plagiarism': {
                                'verdict': 'Plagiarism Detected',
                                'confidence': 0.85,
                                'is_plagiarized': True,
                                'similarity_score': 0.85
                            },
                            'suspicious_file_id': 'uuid-here',
                            'source_file_id': 'uuid-here'
                        }
                    }
                }
            },
            400: {'description': 'Missing required files or invalid file format'},
            500: {'description': 'Server error during plagiarism check'}
        },
        description='Compares two files for plagiarism detection'
    )
    @action(detail=False, methods=['POST'], url_path='file-to-file', url_name='file-to-file', parser_classes=[MultiPartParser, FormParser])
    def file_to_file(self, request, *args, **kwargs):
        try:
            # Extract inputs from request
            suspicious_file = request.FILES.get('suspicious_file')
            source_file = request.FILES.get('source_file')

            # Validate inputs
            if not suspicious_file:
                return Response(
                    {"error": "Suspicious file is required"},
                    status=400
                )

            if not source_file:
                return Response(
                    {"error": "Source file is required"},
                    status=400
                )

            # Validate file types
            suspicious_file_extension = os.path.splitext(suspicious_file.name)[1].lower()
            source_file_extension = os.path.splitext(source_file.name)[1].lower()
            supported_formats = ['.pdf', '.docx', '.txt', '.ipynb']

            if suspicious_file_extension not in supported_formats:
                return Response(
                    {"error": f"Unsupported suspicious file format. Please upload {', '.join(supported_formats)}"},
                    status=400
                )

            if source_file_extension not in supported_formats:
                return Response(
                    {"error": f"Unsupported source file format. Please upload {', '.join(supported_formats)}"},
                    status=400
                )

            # Determine file types
            suspicious_file_type = determine_file_type(suspicious_file_extension)
            source_file_type = determine_file_type(source_file_extension)

            # Save files temporarily to extract content
            temp_suspicious_file = UploadedFile.objects.create(
                file=suspicious_file,
                file_type=suspicious_file_type,
                user=request.user,
                processed=False
            )

            temp_source_file = UploadedFile.objects.create(
                file=source_file,
                file_type=source_file_type,
                user=request.user,
                processed=False
            )

            # Initialize detector
            detector = PlagiarismDetector()

            # Extract text from files
            try:
                suspicious_text = detector.extract_text_from_file(temp_suspicious_file.file.path)
                source_text = detector.extract_text_from_file(temp_source_file.file.path)

                # Check if these contents already exist
                existing_suspicious_content = FileContent.objects.filter(content=suspicious_text).first()
                existing_source_content = FileContent.objects.filter(content=source_text).first()

                # Handle suspicious file
                if existing_suspicious_content:
                    # Use the existing file
                    suspicious_uploaded_file = existing_suspicious_content.uploaded_file
                    # Delete the temporary file since we'll use the existing one
                    temp_suspicious_file.file.delete()
                    temp_suspicious_file.delete()
                else:
                    # Keep the new file
                    suspicious_uploaded_file = temp_suspicious_file
                    # Save extracted content to FileContent model
                    FileContent.objects.create(
                        uploaded_file=suspicious_uploaded_file,
                        content=suspicious_text
                    )

                # Handle source file
                if existing_source_content:
                    # Use the existing file
                    source_uploaded_file = existing_source_content.uploaded_file
                    # Delete the temporary file since we'll use the existing one
                    temp_source_file.file.delete()
                    temp_source_file.delete()
                else:
                    # Keep the new file
                    source_uploaded_file = temp_source_file
                    # Save extracted content to FileContent model
                    FileContent.objects.create(
                        uploaded_file=source_uploaded_file,
                        content=source_text
                    )

                # Mark files as processed
                suspicious_uploaded_file.processed = True
                suspicious_uploaded_file.save()
                source_uploaded_file.processed = True
                source_uploaded_file.save()

                # Check for plagiarism
                result = detector.check_plagiarism(source_text, suspicious_text)

                return Response(
                    {
                        "message": "Plagiarism check successful",
                        "plagiarism": result,
                        "suspicious_file_id": str(suspicious_uploaded_file.guid),
                        "source_file_id": str(source_uploaded_file.guid)
                    },
                    status=200
                )

            except Exception as e:
                # Clean up in case of error
                if hasattr(temp_suspicious_file, 'file'):
                    temp_suspicious_file.file.delete()
                temp_suspicious_file.delete()

                if hasattr(temp_source_file, 'file'):
                    temp_source_file.file.delete()
                temp_source_file.delete()

                logger.error(f"Error extracting text from files: {str(e)}")
                return Response(
                    {"error": f"Failed to extract text from files: {str(e)}"},
                    status=500
                )

        except Exception as e:
            logger.error(f"Error in file-to-file plagiarism detection: {str(e)}")
            return Response(
                {"error": f"Failed to process plagiarism check: {str(e)}"},
                status=500
            )
