# Part 2: clustering and cross-framework concept alignment

Gold: 30 Flask->FastAPI concept pairs (10 seeds for tuning, 20 test) and 7 Flask concepts with no FastAPI counterpart (2 seed, 5 test). All numbers below are on the test items.

## Cluster quality (weak labels = URL section of the page)

| clustering | framework | k | silhouette | sections | purity | NMI | ARI | RI | purity_random | NMI_random |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf/kmeans | flask | 55 | 0.089 | 4 | 0.866 | 0.143 | 0.008 | 0.322 | 0.824 | 0.042 |
| tfidf/kmeans | fastapi | 50 | 0.133 | 10 | 0.737 | 0.439 | 0.141 | 0.826 | 0.313 | 0.040 |
| tfidf/kmeans/centred | flask | 55 | 0.090 | 4 | 0.872 | 0.145 | 0.009 | 0.322 | 0.824 | 0.042 |
| tfidf/kmeans/centred | fastapi | 55 | 0.141 | 10 | 0.739 | 0.434 | 0.131 | 0.826 | 0.316 | 0.044 |
| tfidf/hac | flask | 60 | 0.099 | 4 | 0.896 | 0.178 | 0.017 | 0.331 | 0.824 | 0.045 |
| tfidf/hac | fastapi | 60 | 0.142 | 10 | 0.730 | 0.461 | 0.277 | 0.836 | 0.317 | 0.043 |
| tfidf/hac/centred | flask | 60 | 0.100 | 4 | 0.893 | 0.176 | 0.016 | 0.330 | 0.824 | 0.045 |
| tfidf/hac/centred | fastapi | 60 | 0.151 | 10 | 0.768 | 0.492 | 0.292 | 0.846 | 0.317 | 0.043 |
| lsa/kmeans | flask | 55 | 0.272 | 4 | 0.885 | 0.162 | 0.009 | 0.323 | 0.824 | 0.042 |
| lsa/kmeans | fastapi | 60 | 0.286 | 10 | 0.756 | 0.450 | 0.114 | 0.825 | 0.317 | 0.043 |
| lsa/kmeans/centred | flask | 60 | 0.262 | 4 | 0.875 | 0.156 | 0.008 | 0.321 | 0.824 | 0.045 |
| lsa/kmeans/centred | fastapi | 60 | 0.268 | 10 | 0.750 | 0.440 | 0.129 | 0.827 | 0.317 | 0.043 |
| lsa/hac | flask | 60 | 0.254 | 4 | 0.891 | 0.170 | 0.012 | 0.326 | 0.824 | 0.045 |
| lsa/hac | fastapi | 60 | 0.305 | 10 | 0.780 | 0.496 | 0.281 | 0.846 | 0.317 | 0.043 |
| lsa/hac/centred | flask | 60 | 0.257 | 4 | 0.882 | 0.156 | 0.011 | 0.325 | 0.824 | 0.045 |
| lsa/hac/centred | fastapi | 60 | 0.299 | 10 | 0.781 | 0.497 | 0.282 | 0.846 | 0.317 | 0.043 |
| minilm/kmeans | flask | 35 | 0.146 | 4 | 0.876 | 0.148 | 0.014 | 0.333 | 0.824 | 0.028 |
| minilm/kmeans | fastapi | 60 | 0.178 | 10 | 0.720 | 0.413 | 0.126 | 0.825 | 0.317 | 0.043 |
| minilm/kmeans/centred | flask | 45 | 0.155 | 4 | 0.880 | 0.155 | 0.011 | 0.326 | 0.824 | 0.035 |
| minilm/kmeans/centred | fastapi | 60 | 0.200 | 10 | 0.728 | 0.410 | 0.110 | 0.824 | 0.317 | 0.043 |
| minilm/hac | flask | 55 | 0.086 | 4 | 0.880 | 0.162 | 0.026 | 0.380 | 0.824 | 0.042 |
| minilm/hac | fastapi | 60 | 0.199 | 10 | 0.695 | 0.435 | 0.236 | 0.810 | 0.317 | 0.043 |
| minilm/hac/centred | flask | 55 | 0.157 | 4 | 0.885 | 0.170 | 0.016 | 0.332 | 0.824 | 0.042 |
| minilm/hac/centred | fastapi | 60 | 0.203 | 10 | 0.729 | 0.441 | 0.262 | 0.838 | 0.317 | 0.043 |

## Alignment: Flask concept -> FastAPI chunks

Baselines (plain retrieval into the FastAPI index):

| system | hit@1 | hit@3 | hit@10 | MRR |
|:---|---:|---:|---:|---:|
| BM25F: Flask term -> FastAPI index | 0.500 | 0.650 | 0.700 | 0.579 |
| Dense LSA: Flask term -> FastAPI index | 0.500 | 0.550 | 0.750 | 0.557 |
| Dense MiniLM: Flask term -> FastAPI index | 0.300 | 0.450 | 0.750 | 0.395 |

Cluster-based alignment (m = clusters searched, w = weight of the dense evidence in RRF; tuned on seeds):

| space | method | centred | route | m | w | 1:1 | hit@1 | hit@3 | hit@10 | MRR | hit@3 (no restriction) | MRR (no restriction) | no-match P | no-match R |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | kmeans | no | evidence | 3 | 0.500 | no | 0.550 | 0.750 | 0.800 | 0.656 | 0.800 | 0.671 | 1.000 | 0.200 |
| tfidf | kmeans | yes | evidence | 2 | 1.000 | no | 0.500 | 0.650 | 0.750 | 0.588 | 0.750 | 0.629 | 1.000 | 0.200 |
| tfidf | hac | no | evidence | 2 | 0.500 | no | 0.600 | 0.700 | 0.800 | 0.670 | 0.800 | 0.671 | 1.000 | 0.400 |
| tfidf | hac | yes | evidence | 2 | 1.000 | no | 0.500 | 0.750 | 0.800 | 0.624 | 0.750 | 0.629 | nan | 0.000 |
| lsa | kmeans | no | evidence | 3 | 1.000 | no | 0.450 | 0.700 | 0.850 | 0.596 | 0.700 | 0.585 | 1.000 | 0.400 |
| lsa | kmeans | yes | evidence | 2 | 0.300 | no | 0.450 | 0.550 | 0.700 | 0.528 | 0.700 | 0.602 | 1.000 | 0.200 |
| lsa | hac | no | evidence | 1 | 1.000 | no | 0.600 | 0.700 | 0.700 | 0.650 | 0.700 | 0.585 | 1.000 | 0.200 |
| lsa | hac | yes | evidence | 2 | 0.500 | no | 0.550 | 0.650 | 0.800 | 0.617 | 0.750 | 0.618 | 1.000 | 0.200 |
| minilm | kmeans | no | evidence | 2 | 0.700 | no | 0.450 | 0.600 | 0.750 | 0.547 | 0.700 | 0.588 | nan | 0.000 |
| minilm | kmeans | yes | centroid | 5 | 0.300 | no | 0.400 | 0.500 | 0.750 | 0.487 | 0.650 | 0.568 | 0.250 | 0.200 |
| minilm | hac | no | evidence | 5 | 0.300 | no | 0.450 | 0.700 | 0.850 | 0.570 | 0.650 | 0.617 | 0.000 | 0.000 |
| minilm | hac | yes | evidence | 5 | 0.700 | no | 0.550 | 0.700 | 0.950 | 0.651 | 0.650 | 0.614 | nan | 0.000 |

Significance of the seed-selected configuration vs the baselines (per-pair reciprocal rank, n=20):

| comparison | delta MRR | p (bootstrap) | p (Wilcoxon) |
|:---|---:|---:|---:|
| minilm/kmeans/centred vs BM25F baseline | -0.092 | 0.258 | 0.398 |
| minilm/kmeans/centred vs Dense LSA baseline | -0.070 | 0.507 | 0.574 |

Best configuration on the seeds: **minilm/kmeans/centred** {"m": 5, "w": 0.3, "one_to_one": false, "route": "centroid", "tau": 0.261}

## Cluster pruning: speed vs recall (best configuration)

| clusters searched | docs scored | % of FastAPI | hit@3 | hit@10 | MRR | ms/query |
|:---|---:|---:|---:|---:|---:|---:|
| 1 | 31.450 | 1.548 | 0.350 | 0.550 | 0.389 | 4.168 |
| 2 | 60.950 | 3.000 | 0.400 | 0.650 | 0.422 | 2.738 |
| 3 | 92.200 | 4.537 | 0.500 | 0.750 | 0.489 | 2.677 |
| 5 | 152.900 | 7.525 | 0.500 | 0.750 | 0.487 | 2.922 |
| 10 | 334.400 | 16.457 | 0.600 | 0.800 | 0.545 | 3.186 |
| 20 | 675.700 | 33.253 | 0.650 | 0.850 | 0.566 | 3.962 |
| 60 | 2032.000 | 100.000 | 0.650 | 0.800 | 0.568 | 7.642 |

## Per concept (best configuration, test items)

| pid | flask | expected | sim | gold rank | predicted | matched cluster |
|:---|---:|---:|---:|---:|---:|---:|
| P02 | abort | HTTPException | 0.684 | 7 | fastapi/reference/exceptions#exceptions-httpexception-and-websocketexception | fastapi#100 [exception, httpexception, http, raise] (25 chunks; Handling Errors) |
| P03 | request.args | query parameters | 0.273 | None | fastapi/reference/request#fastapi.Request.form | fastapi#66 [response, responses, streaming, redirect] (72 chunks; Custom Response Classes - File, HTML, Redirect, Streaming, etc.) |
| P05 | request.files | UploadFile | 0.703 | 1 | fastapi/tutorial/request-files#request-files | fastapi#97 [uploadfile, upload, file, files] (25 chunks; UploadFile class) |
| P06 | jsonify | JSONResponse | 0.568 | 5 | fastapi/tutorial/stream-json-lines#json-lines | fastapi#50 [json, pydantic, model, data] (43 chunks; Response Model - Return Type) |
| P08 | before_request | middleware | 0.236 | None | no match | fastapi#92 [query, parameters, validations, parameter] (21 chunks; Request Parameters) |
| P09 | test_client | TestClient | 0.560 | 1 | fastapi/reference/testclient#fastapi.testclient.TestClient.is_closed | fastapi#85 [test, testclient, tests, client] (19 chunks; Async Tests) |
| P11 | send_file | FileResponse | 0.703 | None | fastapi/tutorial/request-files#file-parameters-with-uploadfile | fastapi#97 [uploadfile, upload, file, files] (25 chunks; UploadFile class) |
| P12 | redirect | RedirectResponse | 0.236 | None | no match | fastapi#92 [query, parameters, validations, parameter] (21 chunks; Request Parameters) |
| P14 | set_cookie | response cookies | 0.529 | 1 | fastapi/reference/response#fastapi.Response.set_cookie | fastapi#69 [cookie, lax, samesite, httponly] (37 chunks; Custom Response Classes - File, HTML, Redirect, Streaming, etc.) |
| P15 | request.cookies | Cookie parameters | 0.529 | 4 | fastapi/reference/request#fastapi.Request.cookies | fastapi#69 [cookie, lax, samesite, httponly] (37 chunks; Custom Response Classes - File, HTML, Redirect, Streaming, etc.) |
| P17 | variable rules converters | path parameters | 0.316 | 9 | fastapi/tutorial/bigger-applications#path-operations-with-apirouter | fastapi#65 [router, apirouter, routing.py, fastapi] (38 chunks; Bigger Applications - Multiple Files) |
| P18 | app.config from_envvar | settings | 0.288 | 1 | fastapi/advanced/settings#read-settings-from-env | fastapi#70 [path, operation, predefined, parameters] (21 chunks; Settings and Environment Variables) |
| P20 | stream_with_context streaming | StreamingResponse | 0.511 | 1 | fastapi/advanced/stream-data#a-streamingresponse-with-yield | fastapi#57 [response, fastapi, responses, fast] (32 chunks; Custom Response - HTML, Stream, File, others) |
| P21 | celery background tasks | BackgroundTasks | 0.608 | 1 | fastapi/tutorial/background-tasks#caveat | fastapi#58 [tasks, background, backgroundtasks, notification] (18 chunks; Background Tasks - BackgroundTasks) |
| P23 | flask run development server | fastapi dev | 0.402 | 2 | fastapi/index#run-it | fastapi#73 [concepts, deployments, processes, running] (37 chunks; Deployments Concepts) |
| P24 | login_required decorator | get_current_user dependency | 0.375 | 5 | fastapi/tutorial/request-forms#import-form | fastapi#93 [user, model, password, username] (26 chunks; Form Models) |
| P26 | ProxyFix | behind a proxy | 0.364 | 1 | fastapi/advanced/behind-a-proxy#how-proxy-forwarded-headers-work | fastapi#73 [concepts, deployments, processes, running] (37 chunks; Deployments Concepts) |
| P27 | WSGI middleware | ASGI middleware | 0.364 | None | fastapi/advanced/wsgi#check-it | fastapi#73 [concepts, deployments, processes, running] (37 chunks; Deployments Concepts) |
| P29 | url_for | url_for in templates | 0.236 | 3 | no match | fastapi#92 [query, parameters, validations, parameter] (21 chunks; Request Parameters) |
| P30 | application dispatching | sub applications mount | 0.355 | 1 | fastapi/advanced/sub-applications#sub-application | fastapi#73 [concepts, deployments, processes, running] (37 chunks; Deployments Concepts) |
| N03 | MethodView | (no match) | 0.316 | - | fastapi/advanced/generate-clients#generate-a-typescript-client-with-custom-operation-ids | fastapi#65 [router, apirouter, routing.py, fastapi] (38 chunks; Bigger Applications - Multiple Files) |
| N04 | instance folder | (no match) | 0.558 | - | fastapi/tutorial/static-files#use-staticfiles | fastapi#48 [static, staticfiles, files, templates] (36 chunks; Static Files - StaticFiles) |
| N05 | url_value_preprocessor | (no match) | 0.236 | - | no match | fastapi#92 [query, parameters, validations, parameter] (21 chunks; Request Parameters) |
| N06 | flask shell | (no match) | 0.318 | - | fastapi/deployment/docker#use-cmd-exec-form | fastapi#71 [uvicorn, run, fastapi, span] (34 chunks; FastAPI) |
| N07 | flask.g | (no match) | 0.270 | - | fastapi/advanced/wsgi#check-it | fastapi#94 [httpconnection, request, connection, class] (28 chunks; Request class) |

## Hierarchy

Dendrogram: `concept_dendrogram.png`. First cross-framework merges of two single clusters (HAC finds these alignments without any supervision):

* d=0.297: FL [file, files, send, uploads]  <->  FA [uploadfile, upload, file, files]
* d=0.316: FL [error, exception, handler, exceptions]  <->  FA [exception, httpexception, http, raise]
* d=0.360: FL [flask, application, applications, wsgi]  <->  FA [inspiration, alternatives, comparisons, inspired]
* d=0.384: FL [test, runner, testing, tests]  <->  FA [test, testclient, tests, client]
* d=0.392: FL [celery, task, tasks, result]  <->  FA [tasks, background, backgroundtasks, notification]
* d=0.407: FL [gevent, async, sync, await]  <->  FA [concurrency, await, async, asynchronous]
* d=0.419: FL [db, database, sql, sqlalchemy]  <->  FA [hero, sqlmodel, sql, model]
* d=0.442: FL [folder, static, static_folder, flask]  <->  FA [static, staticfiles, files, templates]
* d=0.489: FL [stream, template, iterator, generator]  <->  FA [response, fastapi, responses, fast]
* d=0.510: FL [teardown, context, request, appcontext]  <->  FA [lifespan, events, shutdown, startup]
* d=0.649: FL [headers, security, policy, cross]  <->  FA [middleware, cors, corsmiddleware, origin]
