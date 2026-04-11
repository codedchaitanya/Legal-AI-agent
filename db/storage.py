# db/storage.py
from minio import Minio
from minio.error import S3Error
from core.config import settings
import io


class StorageClient:
    def __init__(self):
        self._client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )

    def ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    def upload(self, bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.ensure_bucket(bucket)
        self._client.put_object(
            bucket, object_name,
            io.BytesIO(data), length=len(data),
            content_type=content_type,
        )
        return f"{bucket}/{object_name}"

    def download(self, bucket: str, object_name: str) -> bytes:
        response = self._client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


storage = StorageClient()
