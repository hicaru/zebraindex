use anyhow::Result;
use arrow::array::{RecordBatch, RecordBatchIterator};
use lancedb::table::Table;

/// Generic upsert: merges `batch` into `table` on the given key column.
///
/// Shared by FilesTable, ProjectsTable, and ChunksTable to avoid
/// duplicating the same merge-insert boilerplate.
pub async fn upsert_batch(table: &Table, key: &str, batch: RecordBatch) -> Result<()> {
    let schema = batch.schema();
    let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));

    let mut builder = table.merge_insert(&[key]);
    builder.when_matched_update_all(None);
    builder.when_not_matched_insert_all();
    builder.execute(reader).await?;

    Ok(())
}

/// Append `batches` to `table` in a single commit, bypassing the merge-insert
/// scan/optimize machinery.
///
/// Only sound when the caller guarantees none of the incoming keys already
/// exist in the table (e.g. the indexer deletes a file's prior chunks before
/// re-inserting them, so a freshly (re)indexed chunk_id can never collide with
/// a surviving row). One `add` = one manifest commit, regardless of how many
/// batches are bundled — collapsing the per-batch write churn.
pub async fn append_batches(table: &Table, batches: Vec<RecordBatch>) -> Result<()> {
    if batches.is_empty() {
        return Ok(());
    }
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    tracing::info!(
        batches = batches.len(),
        rows,
        "Lance append commit"
    );
    table.add(batches).execute().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn id_batch(ids: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))])
            .expect("build id batch")
    }

    async fn temp_table(name: &str) -> Table {
        let dir = std::env::temp_dir().join(format!("zrag-store-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let db = lancedb::connect(dir.to_str().expect("utf8 path"))
            .execute()
            .await
            .expect("connect");
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        db.create_empty_table("t", schema)
            .execute()
            .await
            .expect("create table")
    }

    #[tokio::test]
    async fn append_batches_bundles_all_rows() {
        let table = temp_table("bundle").await;
        // Two batches in one call land as a single commit, all rows present.
        append_batches(&table, vec![id_batch(&[1, 2]), id_batch(&[3, 4, 5])])
            .await
            .expect("first append");
        assert_eq!(table.count_rows(None).await.expect("count1"), 5);
        // A subsequent append accumulates rather than replacing.
        append_batches(&table, vec![id_batch(&[6])])
            .await
            .expect("second append");
        assert_eq!(table.count_rows(None).await.expect("count2"), 6);
    }

    #[tokio::test]
    async fn append_batches_empty_is_noop() {
        let table = temp_table("empty").await;
        append_batches(&table, Vec::new())
            .await
            .expect("noop append");
        assert_eq!(table.count_rows(None).await.expect("count"), 0);
    }
}
