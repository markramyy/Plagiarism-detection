from celery import shared_task

from core.utils import extract_text_from_file
from file_handling.models import UploadedFile, FileContent

import logging

logger = logging.getLogger(__name__)


@shared_task
def process_file(file_guid):
    try:
        file_instance = UploadedFile.objects.get(guid=file_guid)
        content = extract_text_from_file(file_instance)

        if content:
            FileContent.objects.create(
                uploaded_file=file_instance,
                content=content
            )

            file_instance.processed = True
            file_instance.save()
        else:
            logger.warning(f"No content extracted from file: {file_instance.file.name}")
            # Still mark as processed to avoid retries
            file_instance.processed = True
            file_instance.save()

    except UploadedFile.DoesNotExist:
        logger.info(f"File with guid {file_guid} not found.")

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
