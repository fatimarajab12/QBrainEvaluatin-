-- Enable required extensions
create extension if not exists vector;
create extension if not exists pgcrypto;

-- Projects table used by API services
create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  description text,
  doc_path text,
  created_at timestamptz not null default now()
);

create table if not exists public.features (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  title text not null,
  content text,
  is_ai_generated boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists idx_features_project_id on public.features(project_id);
create index if not exists idx_features_created_at on public.features(created_at desc);

create table if not exists public.test_cases (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  feature_id uuid not null references public.features(id) on delete cascade,
  title text not null,
  steps jsonb not null default '[]'::jsonb,
  expected_result text not null default '',
  is_ai_generated boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists idx_test_cases_project_id on public.test_cases(project_id);
create index if not exists idx_test_cases_feature_id on public.test_cases(feature_id);
create index if not exists idx_test_cases_created_at on public.test_cases(created_at desc);

create table if not exists public.performance_metrics (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  metric_type text not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_performance_metrics_project_type
on public.performance_metrics(project_id, metric_type, created_at desc);

-- Vector table for RAG chunks/embeddings
create table if not exists public.project_vectors (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  content text not null,
  embedding vector(1536) not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_project_vectors_project_id on public.project_vectors(project_id);
create index if not exists idx_project_vectors_created_at on public.project_vectors(created_at desc);
create index if not exists idx_project_vectors_embedding
on public.project_vectors using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create or replace function public.match_project_vectors(
  in_project_id uuid,
  query_embedding vector(1536),
  match_count int default 5,
  min_similarity float default 0.0
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    pv.id,
    pv.content,
    pv.metadata,
    1 - (pv.embedding <=> query_embedding) as similarity
  from public.project_vectors pv
  where pv.project_id = in_project_id
    and (1 - (pv.embedding <=> query_embedding)) >= min_similarity
  order by pv.embedding <=> query_embedding
  limit match_count;
$$;
