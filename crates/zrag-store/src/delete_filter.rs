const DELETE_FILTER_MAX_BYTES: usize = 8 * 1024;
const FILTER_PREFIX: &str = "file_path IN (";

/// Build `file_path IN ('a', 'b', …)` delete predicates, batched so no single
/// predicate exceeds ~8 KiB.
///
/// An `IN` list parses to one flat `InList` expression. A `file_path = 'a' OR
/// file_path = 'b' OR …` chain, by contrast, builds a deeply nested binary
/// expression tree that overflows the query planner's stack once a project has
/// a few hundred changed files — so always emit `IN` lists, never `OR` chains.
pub(crate) fn file_path_delete_filters(paths: &[&str]) -> Vec<String> {
    let mut filters = Vec::with_capacity(1);
    let mut current = String::with_capacity(DELETE_FILTER_MAX_BYTES);
    let mut items = 0usize;
    for path in paths {
        // Worst case: ", " + "'" + every char escaped + "'", plus the ")" close.
        let item_len = ", '".len() + path.len() * 2 + "')".len();
        if items > 0 && current.len().saturating_add(item_len) > DELETE_FILTER_MAX_BYTES {
            current.push(')');
            filters.push(std::mem::take(&mut current));
            current.reserve(DELETE_FILTER_MAX_BYTES);
            items = 0;
        }
        current.push_str(if items == 0 { FILTER_PREFIX } else { ", " });
        current.push('\'');
        push_escaped_sql_string(path, &mut current);
        current.push('\'');
        items += 1;
    }
    if items > 0 {
        current.push(')');
        filters.push(current);
    }
    filters
}

fn push_escaped_sql_string(value: &str, out: &mut String) {
    value.chars().for_each(|ch| {
        if ch == '\'' {
            out.push('\'');
        }
        out.push(ch);
    });
}

#[cfg(test)]
mod tests {
    use super::file_path_delete_filters;

    #[test]
    fn filters_use_exact_path_strings() {
        // Chunks store absolute paths; files store relative. A relative delete
        // filter must not be assumed to match absolute chunk rows.
        let rel = file_path_delete_filters(&["src/foo.rs"]);
        assert_eq!(rel, vec!["file_path IN ('src/foo.rs')"]);

        let abs = file_path_delete_filters(&["/tmp/proj/src/foo.rs"]);
        assert_eq!(abs, vec!["file_path IN ('/tmp/proj/src/foo.rs')"]);
        assert_ne!(rel, abs);
    }

    #[test]
    fn filters_escape_single_quotes() {
        let f = file_path_delete_filters(&["it's.rs"]);
        assert_eq!(f, vec!["file_path IN ('it''s.rs')"]);
    }
}
