-- Add codigo column to companies table
-- Auto-generate codes for existing companies

ALTER TABLE companies ADD COLUMN IF NOT EXISTS codigo VARCHAR UNIQUE;

CREATE INDEX IF NOT EXISTS ix_companies_codigo ON companies (codigo);

-- Backfill existing companies with random 6-char codes
UPDATE companies
SET codigo = UPPER(SUBSTR(MD5(RANDOM()::TEXT), 1, 6))
WHERE codigo IS NULL;
