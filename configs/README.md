# Configuration snapshots

Four RDD-specific YAML files are copied byte-for-byte. Original absolute dataset paths and upstream include paths are preserved. They are not standalone configurations in this staging layout. The test config maps the framework's val_dataloader to the test-split directory.

See [environment notes](../environments/environment_notes.md) for inherited dependencies and local framework modifications. Only RDD-specific config files are included; the original working checkout remains authoritative for inherited configuration.
