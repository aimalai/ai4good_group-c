# Mixed-precision GRU fix

The previous real-data run failed during Lightning sanity validation because the GRU output and destination tensor had different dtypes under automatic mixed precision.

The assignment now casts hidden[-1] to encoded.dtype before inserting it into encoded.

The original model was backed up before patching. The baseline model was not changed.
