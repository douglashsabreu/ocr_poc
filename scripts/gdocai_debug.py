import argparse
import base64
import json
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Google Doc AI output")
    parser.add_argument("file", type=Path, help="Image/PDF to process")
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--mime", default=None)
    parser.add_argument("--mode", choices=["raw", "entities"], default="raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opts = ClientOptions(api_endpoint=f"{args.location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    name = client.processor_path(args.project, args.location, args.processor)

    content = args.file.read_bytes()
    mime_type = args.mime
    if not mime_type:
        import mimetypes

        mime_type, _ = mimetypes.guess_type(args.file.name)
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=content, mime_type=mime_type or "application/octet-stream"),
    )
    response = client.process_document(request=request)
    document = response.document
    print("Document type:", type(document))
    print("Quality scores present?", any(page.image_quality_scores for page in document.pages))
    payload = document._pb.SerializeToString()
    print("Payload size:", len(payload))
    print(json.dumps(json.loads(document.to_json()), indent=2)[:2000])


if __name__ == "__main__":
    main()
