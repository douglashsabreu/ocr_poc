"""
Copyright: Fretefy
Description: Write data to Azure Blob Storage.
"""

import asyncio
import concurrent.futures
import io
from datetime import datetime, timedelta
from typing import Optional

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import (
    BlobClient,
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)


def create_container(container_name, connection_string):
    """
    Creates a container in Azure Blob Storage if it doesn't exist.

    Args:
        container_name (str): The name of the container.
        connection_string (str): The connection string for Azure Blob Storage.

    Returns:
        ContainerClient: An instance of the container client.
    """
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    if not container_client.exists():
        container_client.create_container()

    return container_client


def write_on_cloud(
    local_file_address,
    blob_name,
    container_name,
    connection_string,
    overwrite: bool = False,
    content_type: Optional[str] = None,
):
    """
    Uploads a local file to Azure Blob Storage.

    Args:
        local_file_address (str): The local file path.
        blob_name (str): The name of the blob in the container.
        container_name (str): The name of the container.
        connection_string (str): The connection string for Azure Blob Storage.
        overwrite (bool): Whether to overwrite the existing blob.
        content_type (str, optional): The content type of the file to be uploaded.
    """
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    content_settings = None
    if content_type:
        content_settings = ContentSettings(content_type=content_type)

    with open(local_file_address, "rb") as data:
        blob_client.upload_blob(
            data, overwrite=overwrite, content_settings=content_settings
        )


def upload_data_sync(
    data_to_upload,
    blob_name,
    container_name,
    connection_string,
    storage_account_name,
    storage_account_key,
    ttl_days: float = 7.0,
):
    """
    Synchronously uploads data (e.g., audio) to Azure Blob Storage.

    Args:
        data_to_upload (AudioSegment): The data to upload (in this case, audio).
        blob_name (str): The name of the blob in the container.
        container_name (str): The name of the container.
        connection_string (str): The connection string for Azure Blob Storage.
        storage_account_name (str): The Azure storage account name.
        storage_account_key (str): The Azure storage account key.
        ttl_days (float): The number of days for the SAS URL to be valid.

    Returns:
        str: A SAS URL with a 7-day expiration for accessing the uploaded blob.
    """
    # Convert AudioSegment to bytes
    audio_buffer = io.BytesIO()
    data_to_upload.export(audio_buffer, format="mp3")
    audio_buffer.seek(0)

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    # Set content settings for the audio file
    content_settings = ContentSettings(content_type="audio/mpeg")

    # Upload the audio data as bytes
    blob_client.upload_blob(
        audio_buffer,
        blob_type="BlockBlob",
        content_settings=content_settings,
        overwrite=True,
    )

    accessible_address = asyncio.run(
        get_address(
            storage_account_name,
            container_name,
            blob_name,
            storage_account_key,
            ttl_days,
        )
    )
    return accessible_address


async def upload_data(
    data_to_upload,
    blob_name,
    container_name,
    connection_string,
    storage_account_name,
    storage_account_key,
    ttl_days: float = 7.0,
):
    """
    Asynchronously uploads data (e.g., audio) to Azure Blob Storage.

    Args:
        data_to_upload (AudioSegment): The data to upload (in this case, audio).
        blob_name (str): The name of the blob in the container.
        container_name (str): The name of the container.
        connection_string (str): The connection string for Azure Blob Storage.
        storage_account_name (str): The Azure storage account name.
        storage_account_key (str): The Azure storage account key.
        ttl_days (float): The number of days for the SAS URL to be valid.

    Returns:
        str: A SAS URL with a 7-day expiration for accessing the uploaded blob.
    """
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            upload_data_sync,
            data_to_upload,
            blob_name,
            container_name,
            connection_string,
            storage_account_name,
            storage_account_key,
            ttl_days,
        )
    return result


async def get_address(
    storage_account_name,
    container_name,
    blob_name,
    storage_account_key,
    ttl_days: float = 7.0,
):
    """
    Generates a SAS URL for a blob, granting read access for a default of 7 days.

    Args:
        storage_account_name (str): The Azure storage account name.
        container_name (str): The name of the container.
        blob_name (str): The name of the blob in the container.
        storage_account_key (str): The Azure storage account key.
        ttl_days (float): The number of days for the SAS URL to be valid.

    Returns:
        str: A SAS URL with a 7-day expiration for accessing the blob.
    """
    sas_token = generate_blob_sas(
        account_name=storage_account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=storage_account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(days=ttl_days),
    )
    url_with_sas = (
        f"https://{storage_account_name}.blob.core.windows.net/"
        f"{container_name}/{blob_name}?{sas_token}"
    )

    return url_with_sas


def write_on_cloud_get_address(
    local_file_address,
    blob_name,
    container_name,
    connection_string,
    storage_account_name,
    storage_account_key,
    overwrite: bool = False,
    ttl_days: float = 7.0,
    content_type: Optional[str] = None,
):
    """
    Uploads a file to Azure Blob Storage and returns a SAS URL for accessing the uploaded file.

    Args:
        local_file_address (str): The local file path.
        blob_name (str): The name of the blob in the container.
        container_name (str): The name of the container.
        connection_string (str): The connection string for Azure Blob Storage.
        storage_account_name (str): The Azure storage account name.
        storage_account_key (str): The Azure storage account key.
        overwrite (bool): Whether to overwrite the existing blob.
        ttl_days (float): The number of days for the SAS URL to be valid.
        content_type (str, optional): The content type of the file to be uploaded.

    Returns:
        str: A SAS URL with a 7-day expiration for accessing the uploaded blob.
    """
    create_container(container_name, connection_string)

    write_on_cloud(
        local_file_address,
        blob_name,
        container_name,
        connection_string,
        overwrite=overwrite,
        content_type=content_type,
    )

    accessible_address = asyncio.run(
        get_address(
            storage_account_name,
            container_name,
            blob_name,
            storage_account_key,
            ttl_days,
        )
    )

    return accessible_address


def get_blob_url_if_exists(
    connection_string: str,
    container_name: str,
    blob_name: str,
    account_name: str,
    account_key: str,
) -> str:
    """
    Check if blob exists and return its URL if it does.

    Args:
        connection_string: Azure Storage connection string
        container_name: Name of the container
        blob_name: Name of the blob to check
        account_name: Storage account name
        account_key: Storage account key

    Returns:
        str: URL of the blob if it exists, empty string otherwise
    """
    blob_client = BlobClient.from_connection_string(
        conn_str=connection_string, container_name=container_name, blob_name=blob_name
    )

    try:
        blob_client.get_blob_properties()
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1),
        )

        # Construct the full URL
        blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        return blob_url

    except ResourceNotFoundError:
        return ""


def upload_bytes_get_address(
    data_bytes: bytes,
    blob_name: str,
    container_name: str,
    storage_account_name: str,
    storage_account_key: str,
    ttl_days: float = 7.0,
    content_type: Optional[str] = None,
) -> str:
    """
    Uploads bytes data to Azure Blob Storage and returns a SAS URL for accessing the uploaded file.

    Args:
        data_bytes: The bytes data to upload
        blob_name: The name of the blob in the container
        container_name: The name of the container
        storage_account_name: The Azure storage account name
        storage_account_key: The Azure storage account key
        ttl_days: The number of days for the SAS URL to be valid
        content_type: The content type of the data to be uploaded

    Returns:
        str: A SAS URL with expiration for accessing the uploaded blob
    """
    connection_string = generate_connection_string(
        storage_account_name, storage_account_key
    )

    # Create container if it doesn't exist
    create_container(container_name, connection_string)

    # Create blob service client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    # Set content settings if content type is provided
    content_settings = None
    if content_type:
        content_settings = ContentSettings(content_type=content_type)

    # Upload the bytes data
    blob_client.upload_blob(
        data_bytes,
        blob_type="BlockBlob",
        content_settings=content_settings,
        overwrite=True,
    )

    # Generate and return SAS URL
    accessible_address = asyncio.run(
        get_address(
            storage_account_name,
            container_name,
            blob_name,
            storage_account_key,
            ttl_days,
        )
    )

    return accessible_address


def generate_connection_string(
    storage_account_name: str, storage_account_key: str
) -> str:
    """
    Generates an Azure Storage connection string using the provided credentials.

    Args:
        storage_account_name (str): The name of the Azure Storage account
        storage_account_key (str): The access key for the Azure Storage account

    Returns:
        str: A formatted connection string for Azure Storage authentication
    """
    return (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={storage_account_name};"
        f"AccountKey={storage_account_key};"
        f"EndpointSuffix=core.windows.net"
    )
