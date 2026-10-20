"""Local hybrid retrieval: SQLite FTS5 + optional sentence-transformer embeddings + symbol index."""
import ast, hashlib, json, math, re, sqlite3
from pathlib import Path

MODEL_NAME='sentence-transformers/all-MiniLM-L6-v2'
_model=None

def model():
 global _model
 if _model is None:
  try:
   from sentence_transformers import SentenceTransformer
  except ImportError as exc:
   raise RuntimeError('Install optional semantic dependencies: pip install -r requirements-semantic.txt') from exc
  _model=SentenceTransformer(MODEL_NAME, device='cpu')
 return _model

def initialize(db):
 with sqlite3.connect(db) as c:
  c.execute('CREATE TABLE IF NOT EXISTS vectors(chunk_id INTEGER PRIMARY KEY, model TEXT NOT NULL, embedding TEXT NOT NULL)')
  c.execute('CREATE TABLE IF NOT EXISTS symbols(id INTEGER PRIMARY KEY, file_id TEXT, path TEXT, name TEXT, kind TEXT, line INTEGER, signature TEXT)')
  c.execute('CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name)')
  c.execute('CREATE INDEX IF NOT EXISTS symbols_file ON symbols(file_id)')

def vectorize(texts):
 return model().encode(texts, normalize_embeddings=True, batch_size=24, show_progress_bar=False).tolist()

def index_vectors(db, file_id=None):
 initialize(db)
 with sqlite3.connect(db) as c:
  c.row_factory=sqlite3.Row
  sql='SELECT id,content FROM chunks WHERE id NOT IN (SELECT chunk_id FROM vectors WHERE model=?)'
  args=[MODEL_NAME]
  if file_id:sql+=' AND file_id=?';args.append(file_id)
  rows=c.execute(sql,args).fetchall()
  for offset in range(0,len(rows),48):
   batch=rows[offset:offset+48]
   embeddings=vectorize([r['content'] for r in batch])
   c.executemany('INSERT OR REPLACE INTO vectors VALUES(?,?,?)',[(r['id'],MODEL_NAME,json.dumps(v)) for r,v in zip(batch,embeddings)])
  return {'indexed_vectors':len(rows),'model':MODEL_NAME}

def semantic_search(db,query,file_ids=None,limit=10):
 initialize(db)
 q=vectorize([query])[0]
 with sqlite3.connect(db) as c:
  c.row_factory=sqlite3.Row
  sql='SELECT chunks.file_id,chunks.path,chunks.ordinal,chunks.content,vectors.embedding FROM vectors JOIN chunks ON chunks.id=vectors.chunk_id WHERE vectors.model=?'
  args=[MODEL_NAME]
  if file_ids:sql+=' AND chunks.file_id IN ('+','.join('?' for _ in file_ids)+')';args+=file_ids
  results=[]
  for r in c.execute(sql,args):
   v=json.loads(r['embedding']);score=sum(a*b for a,b in zip(q,v))
   results.append({'file_id':r['file_id'],'path':r['path'],'ordinal':r['ordinal'],'content':r['content'],'score':round(score,5),'method':'semantic'})
 return sorted(results,key=lambda r:r['score'],reverse=True)[:limit]

def hybrid_search(db,query,keyword_results,file_ids=None,limit=12):
 # Reciprocal rank fusion gives comparable rankings without assuming scores share a scale.
 try:semantic=semantic_search(db,query,file_ids,limit*3)
 except RuntimeError:semantic=[]
 rankings={}
 for source,results in [('keyword',keyword_results),('semantic',semantic)]:
  for rank,r in enumerate(results):
   key=(r['file_id'],r['path'],r['ordinal'])
   if key not in rankings:rankings[key]={'item':dict(r),'score':0,'methods':[]}
   rankings[key]['score']+=1/(60+rank+1)
   rankings[key]['methods'].append(source)
 out=[]
 for x in sorted(rankings.values(),key=lambda x:x['score'],reverse=True)[:limit]:
  item=x['item'];item['rank_score']=round(x['score'],6);item['methods']=x['methods'];out.append(item)
 return out

def index_symbols(c,fid,path,body):
 if path.endswith('.py'):
  try:
   tree=ast.parse(body)
   for node in ast.walk(tree):
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
     kind='class' if isinstance(node,ast.ClassDef) else 'function'
     signature=body.splitlines()[node.lineno-1].strip()[:400]
     c.execute('INSERT INTO symbols(file_id,path,name,kind,line,signature) VALUES(?,?,?,?,?,?)',(fid,path,node.name,kind,node.lineno,signature))
  except SyntaxError:pass
 elif Path(path).suffix.lower() in {'.js','.jsx','.ts','.tsx','.go','.rs','.java'}:
  for i,line in enumerate(body.splitlines(),1):
   m=re.search(r'\b(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|struct|enum|func)\s+([A-Za-z_$][\w$]*)',line)
   if m:c.execute('INSERT INTO symbols(file_id,path,name,kind,line,signature) VALUES(?,?,?,?,?,?)',(fid,path,m.group(1),'declaration',i,line.strip()[:400]))

def search_symbols(db,q,file_id=None,limit=40):
 initialize(db)
 with sqlite3.connect(db) as c:
  c.row_factory=sqlite3.Row
  sql='SELECT file_id,path,name,kind,line,signature FROM symbols WHERE name LIKE ? ESCAPE "\\"';args=['%'+q.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%']
  if file_id:sql+=' AND file_id=?';args.append(file_id)
  sql+=' ORDER BY name LIMIT ?';args.append(min(limit,100))
  return [dict(r) for r in c.execute(sql,args)]
