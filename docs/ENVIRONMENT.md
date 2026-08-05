# Environment

## JDK pin (critical)

Spark 4.x supports **Java 17/21 only**. This machine's default `java` is
**25.0.2** (Homebrew's unversioned `openjdk` is first on `PATH` via
`~/.zprofile`) — running Spark under it risks JVM module-access/reflection
failures whose error messages do not point at the version as the cause.
`openjdk@21` (21.0.12) is installed but keg-only, so `/usr/libexec/java_home`
does not see it by default.

**The global default is never changed.** The pin is applied project-locally,
in both variables — since PySpark reads `JAVA_HOME` while other tools resolve
`java` via `PATH`:

```
JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
PATH="$JAVA_HOME/bin:$PATH"
```

These are set in the `Makefile` (exported, so every `make` target inherits
them) and the same version is pinned in CI. Outside of `make`-invoked
commands (i.e. a plain shell), `java -version` continues to report the host
default (25.x) — this is expected and correct; only the project env is
pinned.

**Troubleshooting:** if Spark fails to start with opaque JVM module/reflection
errors, check `java -version` inside the project env (`make java-check`)
first.

## Python / package management

- Python 3.12, managed via `uv` (`uv sync --dev`; run everything as
  `uv run …`).
- `pyspark==4.0.4`.
- Iceberg runtime jar: `org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.11.0`,
  fetched via `spark.jars.packages` (not vendored).
