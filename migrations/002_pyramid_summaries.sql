-- Rename document summary columns to pyramid naming
ALTER TABLE documents RENAME COLUMN summary_l1 TO summary_8w;
ALTER TABLE documents RENAME COLUMN summary_l2 TO summary_32w;
ALTER TABLE documents RENAME COLUMN summary_l3 TO summary_128w;

-- Add new intermediate document summary levels
ALTER TABLE documents ADD COLUMN summary_16w TEXT;
ALTER TABLE documents ADD COLUMN summary_64w TEXT;

-- Rename section summary columns to pyramid naming
-- section_summary_l2 was the short phrase (~8 words), rename first
ALTER TABLE sections RENAME COLUMN section_summary_l2 TO section_summary_8w;
-- section_summary was the 1-2 sentence (~32 words), rename second
ALTER TABLE sections RENAME COLUMN section_summary TO section_summary_32w;

-- Add new section summary level
ALTER TABLE sections ADD COLUMN section_summary_128w TEXT;
