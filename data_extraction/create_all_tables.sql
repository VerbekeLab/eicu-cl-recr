-- creates all the tables and produces csv files
-- takes a few minutes to run

-- \i /Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/data-extraction/labels.sql
-- \i /Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/data-extraction/flat_features.sql
-- \i /Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/data-extraction/timeseries.sql


\i /mnt/tmp-eicu/data-extraction/code/labels.sql
\i /mnt/tmp-eicu/data-extraction/code/flat_features.sql
\i /mnt/tmp-eicu/data-extraction/code/timeseries.sql

-- we need to make sure that we have at least some form of time series for every patient in diagnoses, flat and labels
drop materialized view if exists timeseries_patients cascade;
create materialized view timeseries_patients as
  with repeats as (
    select distinct patientunitstayid
      from timeserieslab
    union
    select distinct patientunitstayid
      from timeseriesresp
    union
    select distinct patientunitstayid
      from timeseriesperiodic
    union
    select distinct patientunitstayid
      from timeseriesaperiodic)
  select distinct patientunitstayid
    from repeats;

\copy (select * from labels as l where l.patientunitstayid in (select * from timeseries_patients)) to '/mnt/tmp-eicu/eICU_data/labels.csv' with csv header
\copy (select * from flat as f where f.patientunitstayid in (select * from timeseries_patients)) to '/mnt/tmp-eicu/eICU_data/flat_features.csv' with csv header
\copy (select * from timeserieslab) to '/mnt/tmp-eicu/eICU_data/timeserieslab.csv' with csv header
\copy (select * from timeseriesresp) to '/mnt/tmp-eicu/eICU_data/timeseriesresp.csv' with csv header
\copy (select * from timeseriesperiodic) to '/mnt/tmp-eicu/eICU_data/timeseriesperiodic.csv' with csv header
\copy (select * from timeseriesaperiodic) to '/mnt/tmp-eicu/eICU_data/timeseriesaperiodic.csv' with csv header

-- \copy (select * from labels as l where l.patientunitstayid in (select * from timeseries_patients)) to '/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_data/labels.csv' with csv header
-- \copy (select * from flat as f where f.patientunitstayid in (select * from timeseries_patients)) to '/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_data/flat_features.csv' with csv header
-- \copy (select * from timeserieslab) to '/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_data/timeserieslab.csv' with csv header
-- \copy (select * from timeseriesresp) to '/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_data/timeseriesresp.csv' with csv header
-- \copy (select * from timeseriesperiodic) to '/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_data/timeseriesperiodic.csv' with csv header
-- \copy (select * from timeseriesaperiodic) to '/Users/vscheltjens/Documents/PhD/Projects/eicu-cl-req/eICU_data/timeseriesaperiodic.csv' with csv header