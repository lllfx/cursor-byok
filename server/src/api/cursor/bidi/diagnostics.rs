//! Summarizes wire boundaries without logging message values or file contents.
use prost::encoding::{decode_key, decode_varint, skip_field, DecodeContext, WireType};
use serde_json::{json, Value};

const MAX_FIELDS: usize = 96;

pub(super) fn wire_structure(payload: &[u8]) -> Value {
    let mut fields = Vec::new();
    scan(payload, 0, &[], &mut fields);
    json!({ "fields": fields, "limit": MAX_FIELDS, "limit_reached": fields.len() == MAX_FIELDS })
}

fn scan(mut bytes: &[u8], base: usize, path: &[u32], fields: &mut Vec<Value>) {
    let size = bytes.len();
    while !bytes.is_empty() {
        let offset = base + size - bytes.len();
        let (tag, wire) = match decode_key(&mut bytes) {
            Ok(key) => key,
            Err(error) => {
                record(
                    json!({ "path": path, "offset": offset, "error": error.to_string() }),
                    fields,
                );
                return;
            }
        };
        let mut field_path = path.to_vec();
        field_path.push(tag);
        let mut field = json!({ "path": field_path, "offset": offset, "wire_type": wire as u8 });
        if wire == WireType::LengthDelimited {
            let length = match decode_varint(&mut bytes) {
                Ok(length) => length,
                Err(error) => {
                    field["error"] = error.to_string().into();
                    record(field, fields);
                    return;
                }
            };
            field["length"] = length.into();
            if length > bytes.len() as u64 {
                field["error"] = "length exceeds remaining bytes".into();
                field["remaining_bytes"] = bytes.len().into();
                record(field, fields);
                return;
            }
            let (value, rest) = bytes.split_at(length as usize);
            let value_offset = base + size - bytes.len();
            record(field, fields);
            // AgentClientMessage.run_request → conversation_state → file_states map entry.
            // Map keys and values are opaque: never interpret them as nested messages.
            if matches!(field_path.as_slice(), [1] | [1, 1] | [1, 1, 12]) {
                scan(value, value_offset, &field_path, fields);
            }
            bytes = rest;
        } else {
            let result = skip_field(wire, tag, &mut bytes, DecodeContext::default());
            if let Err(error) = result {
                field["error"] = error.to_string().into();
                record(field, fields);
                return;
            }
            record(field, fields);
        }
    }
}

fn record(field: Value, fields: &mut Vec<Value>) {
    if fields.len() < MAX_FIELDS {
        fields.push(field);
    } else if field.get("error").is_some() {
        // Still locate failures when a large conversation exceeds the log output limit.
        fields[MAX_FIELDS - 1] = field;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(tag: u32, value: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        prost::encoding::encode_key(tag, WireType::LengthDelimited, &mut bytes);
        prost::encoding::encode_varint(value.len() as u64, &mut bytes);
        bytes.extend_from_slice(value);
        bytes
    }

    #[test]
    fn identifies_invalid_map_wire_type_at_absolute_offset() {
        let payload = field(1, &field(1, &field(12, &[6])));
        let report = wire_structure(&payload);
        let error = report["fields"].as_array().unwrap().last().unwrap();
        assert_eq!(error["path"], json!([1, 1, 12]));
        assert_eq!(error["offset"], payload.len() - 1);
        assert!(error["error"]
            .as_str()
            .unwrap()
            .contains("invalid wire type"));
    }

    #[test]
    fn reports_map_boundaries_without_logging_keys_or_values() {
        let mut entry = field(1, b"private-file-name");
        entry.extend(field(2, b"secret-file-content"));
        let payload = field(1, &field(1, &field(12, &entry)));
        let report = wire_structure(&payload);
        assert_eq!(report["fields"][3]["path"], json!([1, 1, 12, 1]));
        assert_eq!(report["fields"][4]["path"], json!([1, 1, 12, 2]));
        let text = report.to_string();
        assert!(!text.contains("private-file-name"));
        assert!(!text.contains("secret-file-content"));
        assert!(!text.contains("error"));
    }

    #[test]
    fn bounds_output_and_handles_truncated_lengths() {
        let payload = [8, 0].repeat(MAX_FIELDS + 1);
        let report = wire_structure(&payload);
        assert_eq!(report["fields"].as_array().unwrap().len(), MAX_FIELDS);
        assert_eq!(report["limit_reached"], true);
        let report = wire_structure(&[10, 127]);
        assert_eq!(
            report["fields"][0]["error"],
            "length exceeds remaining bytes"
        );
        assert!(wire_structure(&[10])["fields"][0]["error"].is_string());
    }

    #[test]
    fn locates_map_failure_after_many_conversation_turns() {
        let mut state = field(8, b"opaque-turn-id").repeat(MAX_FIELDS + 1);
        state.extend(field(12, &[6]));
        let payload = field(1, &field(1, &state));
        let report = wire_structure(&payload);
        let error = report["fields"].as_array().unwrap().last().unwrap();
        assert_eq!(error["path"], json!([1, 1, 12]));
        assert_eq!(error["offset"], payload.len() - 1);
        assert!(error["error"].is_string());
    }
}
