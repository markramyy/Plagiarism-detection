from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny

from core.base_viewset import BaseViewSet
from plagiarism_model.services.testing import PlagiarismDetector

import logging
from drf_spectacular.utils import extend_schema

logger = logging.getLogger(__name__)


class PlagiarismViewSet(BaseViewSet):
    permission_classes = [AllowAny]

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

    @action(detail=False, methods=['POST'], url_path='text-to-file', url_name='text-to-file')
    def text_to_file(self, request, *args, **kwargs):
        pass

    @action(detail=False, methods=['POST'], url_path='file-to-file', url_name='file-to-file')
    def file_to_file(self, request, *args, **kwargs):
        pass
