import os
import json
import tempfile
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .rag_engine.pipeline import RAGPipeline

# Singleton RAG Pipeline instance
_RAG_PIPELINE = None

def get_pipeline():
    global _RAG_PIPELINE
    if _RAG_PIPELINE is None:
        db_dir = getattr(settings, 'CHROMA_DB_DIR', os.path.join(settings.BASE_DIR, 'data', 'chroma_db'))
        _RAG_PIPELINE = RAGPipeline(db_dir=db_dir)
    return _RAG_PIPELINE


def index(request):
    """Render the main chat web interface."""
    pipeline = get_pipeline()
    stats = pipeline.get_system_stats()
    context = {
        'stats': stats,
        'gemini_configured': stats.get('gemini_api_configured', False)
    }
    return render(request, 'index.html', context)


@csrf_exempt
def api_query(request):
    """API endpoint to process RAG chat query."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed. Use POST.'}, status=405)

    try:
        body = json.loads(request.body.decode('utf-8'))
        query_text = body.get('query', '').strip()
        n_results = int(body.get('n_results', 4))

        if not query_text:
            return JsonResponse({'error': 'Query text cannot be empty.'}, status=400)

        pipeline = get_pipeline()
        result = pipeline.query(query_text=query_text, n_results=n_results)
        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON body.'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def api_ingest(request):
    """API endpoint to ingest text or file into RAG knowledge base."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed. Use POST.'}, status=405)

    try:
        pipeline = get_pipeline()

        # Handle uploaded file
        if request.FILES and 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            filename = uploaded_file.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            try:
                res = pipeline.ingest_file(tmp_path, source_name=filename)
                return JsonResponse(res)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Handle JSON body text ingestion
        elif request.content_type == 'application/json':
            body = json.loads(request.body.decode('utf-8'))
            text = body.get('text', '').strip()
            source_name = body.get('source_name', 'pasted_text').strip()

            if not text:
                return JsonResponse({'error': 'No text provided for ingestion.'}, status=400)

            res = pipeline.ingest_text(text, source_name=source_name)
            return JsonResponse(res)

        # Handle form data text ingestion
        elif request.POST.get('text'):
            text = request.POST.get('text', '').strip()
            source_name = request.POST.get('source_name', 'form_input').strip()
            res = pipeline.ingest_text(text, source_name=source_name)
            return JsonResponse(res)

        else:
            return JsonResponse({'error': 'No text or file attached in request.'}, status=400)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def api_stats(request):
    """API endpoint returning system & vector database stats."""
    pipeline = get_pipeline()
    stats = pipeline.get_system_stats()
    return JsonResponse(stats)


@csrf_exempt
def api_clear(request):
    """API endpoint to reset the vector database."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed. Use POST.'}, status=405)

    pipeline = get_pipeline()
    res = pipeline.clear_knowledge_base()
    return JsonResponse(res)
