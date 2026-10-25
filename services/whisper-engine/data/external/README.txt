# Internet / external meeting data for fine-tuning
#
# Option A — one folder per meeting:
#   data/external/meeting-01/
#     audio.wav
#     transcript.txt
#     language.txt          (optional: fr or en)
#
# Option B — JSON manifest per meeting:
#   data/external/meeting-02/
#     audio.mp3
#     manifest.json         {"transcript": "...", "language": "fr", "segments": [...]}
#
# Option C — bulk manifest:
#   data/external/manifest.jsonl
#   each line: {"audio": "C:/path/to/audio.wav", "text": "...", "language": "fr"}
#
# IMPORTANT: Never put your evaluation test set here.
# Fixed test meetings live only in data/test_set/.
