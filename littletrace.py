from argparse import ArgumentParser
from dataclasses import dataclass, field, fields
from pathlib import Path
import itertools
import math
import json

###########################################################################################################
# Script main
###########################################################################################################

def main():
    parser = ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--parse", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = TraceConfig(**json.loads(args.config.read_text()))

    if args.generate:
        generate(args, config)
    if args.parse:
        parse(args, config)

def generate(args, config):
    code_generator = CodeGenerator(config)

    hfile = code_generator.hfile_text()
    cfile = code_generator.cfile_text()
    testfile = code_generator.testfile_text()
    freertos_config_text = code_generator.freertos_config_text()

    output_dir = args.config.parent / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    h_file_path = output_dir / config.filename_header
    c_file_path = output_dir / config.filename_source
    test_file_path = output_dir / config.filename_test
    freertos_header_path = output_dir / config.filename_freertos_header

    h_file_path.write_text(hfile)
    if not config.header_only:
        c_file_path.write_text(cfile)
    if config.enable_test:
        test_file_path.write_text(testfile)
    if config.freertos_enable:
        freertos_header_path.write_text(freertos_config_text)

def parse(args, config):
    data = args.parse.read_bytes()
    event_structs = BinaryParser(data, config).get_all()
    for struct in event_structs:
        struct = config.parse_struct(struct)
        print(struct)
    return event_structs

###########################################################################################################
# Configuration
###########################################################################################################

@dataclass
class TraceConfig:
    function_name_begin_update: str = field(metadata={
        "description": (
            "Will be called before adding an entry to the trace.",
            "Typically this should start a critical section." ,
        ),
        "example": '"taskENTER_CRITICAL"',
        "group": "HOOKS",
    })
    function_name_end_update: str = field(metadata={
        "description": (
            "Will be called after adding an entry to the trace.",
            "Typically this should end a critical section.",
        ),
        "example": '"taskEXIT_CRITICAL"',
        "group": "HOOKS",
    })
    data_members: dict = field(metadata={
        "description": (
            "Data type and name of the fields in each trace entry.",
            "Allowed types are uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, and the corresponding pointer types.",
            "If a field is specified with a pointer type, that value will be passed by reference to the function that updates the trace.",
        ),
        "example": '{ "u8": "uint8_t", "u16": "uint16_t", "s32": "int32_t" }',
        "group": "DATA"
    })
    definition_implementation_guard: str = field(metadata={
        "description": (
            "Only used if header_only is set to true.",
            "This is the name of a macro that should be defined in exactly one translation unit, which will instantiate the implementation code.",
            "This follows the pattern in the STB libraries, see: https://github.com/nothings/stb",
        ),
        "example": '"LITTLETRACE_IMPLEMENTATION"',
        "group": "API"
    })
    definition_include_guard: str = field(metadata={
        "description": (
            "The header guard (#ifdef/#define pair) that wraps the header file.",
        ),
        "example": '"__LITTLETRACE_IMPLEMENTATION_H"',
        "group": "API"
    })
    definition_n_entries: str = field(metadata={
        "description": (
            "The name of a macro for the number of entries in the trace buffer. ",
            "The actual number is set in the 'n_entries' field.",
        ),
        "example": "TRACE_N_ENTRIES",
        "group": "API"
    })
    enable_counter: bool = field(metadata={
        "description": (
            "Whether to enable event counting.",
            "The trace may rapidly exceed the capacity of the buffer, so it may be useful to know how many events have occurred.",
        ),
        "example": "true",
        "group": "DATA"
    })
    enable_test: bool = field(metadata={
        "description": (
            "Whether to generate a test file.",
        ),
        "example": "false",
        "group": "TESTING"
    })
    enable_timestamp: bool = field(metadata={
        "description": (
            "Whether to enable timestamping of events.",
        ),
        "example": "true",
        "group": "DATA"
    })
    filename_header: str = field(metadata={
        "description": (
            "The name of the header file for the trace library.",
        ),
        "example": '"littletrace.h"',
        "group": "API"
    })
    filename_source: str = field(metadata={
        "description": (
            "The name of the source file for the trace library. Ignored if header_only is true.",
        ),
        "example": '"littletrace.c"',
        "group": "API"
    })
    filename_test: str = field(metadata={
        "description": (
            "The name of the test file for the trace library. Only used if enable_test is true.",
        ),
        "example": '"littletrace_test.c"',
        "group": "TESTING"
    })
    function_name_get_timestamp: str = field(metadata={
        "description": (
            "Only used if enable_timestamp is set to true.",
            "This is the name of the function that returns a timestamp.",
            "This function is provided externally by the user.",
            "It should take no arguments and return a value of the type set in the 'typename_timestamp' field.",
        ),
        "example": '"trace_get_timestamp"',
        "group": "HOOKS",
    })
    function_name_init_trace: str = field(metadata={
        "description": (
            "The name of the function that initializes the trace.",
        ),
        "example": '"trace_init"',
        "group": "API",
    })
    function_name_update_trace: str = field(metadata={
        "description": (
            "The name of the function that updates the trace.",
        ),
        "example": '"trace_update"',
        "group": "API",
    })
    function_name_test_main: str = field(metadata={
        "description": (
            "The name of the function that executes the tests.",
            "Only used if enable_test is true.",
            "Use 'main' if you want a standalone test file, ",
            "or some other name if you want to incorporate the test file into something else.",
        ),
        "example": '"main"',
        "group": "TESTING",
    })
    function_name_assert: str = field(metadata={
        "description": (
            "The name of the function or macro that performs assertions.",
            "The default 'assert' should work, but you can also use a custom function or macro.",
       ),
        "example": '"assert"',
        "group": "TESTING",
    })
    header_only: bool = field(metadata={
        "description": (
            "If set to true, the implementation of the library will be included in the header file,",
            "wrapped in an #ifdef guard with the name set in the 'definition_implementation_guard' field.",
            "This follows the pattern in the STB libraries, see: https://github.com/nothings/stb",
        ),
        "example": "true",
        "group": "API",
    })
    n_entries: int = field(metadata={
        "description": (
            "The number of entries in the trace buffer. Powers of two are optimal.",
        ),
        "example": "256",
        "group": "DATA",
    })
    output_dir: str = field(metadata={
        "description": (
            "Where to place the generated files, relative to the config file.",
        ),
        "example": '"generated"',
        "group": "API",
    })
    pre_include: str = field(metadata={
        "description": (
            "If set, will be included from the header file. Here you might provide declarations needed by the library",
        ),
        "example": '"trace_config.h"',
        "group": "API",
    })
    typename_counter: str = field(metadata={
        "description": (
            "The datatype of the counter. Only used if enable_counter is true.",
        ),
        "example": '"uint32_t"',
        "group": "DATA"
    })
    typename_event_id: str = field(metadata={
        "description": (
            "The datatype of the event ID.",
        ),
        "example": '"uint8_t"',
        "group": "DATA"
    })
    typename_timestamp: str = field(metadata={
        "description": (
            "The datatype of the timestamps. Only used if enable_timestamp is true.",
        ),
        "example": '"uint32_t"',
        "group": "DATA"
    })
    typename_trace_buffer: str = field(metadata={
        "description": (
            "The name of the datatype for the trace buffer.",
        ),
        "example": '"trace_buffer_t"',
        "group": "API"
    })
    typename_trace_entry: str = field(metadata={
        "description": (
            "The name of the datatype for the trace entries.",
        ),
        "example": '"trace_entry_t"',
        "group": "API"
    })

    freertos_enable: bool = field(metadata={
        "description": (
            "Whether to generate FreeRTOS integration code.",
        ),
        "example": '"true"',
        "group": "FREERTOS",
    })

    freertos_default_args: dict = field(metadata={
        "description": (
            "Default argument mapping for FreeRTOS hooks"
        ),
        "example": "See example config",
        "group": "FREERTOS"
    })

    freertos_hooks: dict = field(metadata={
        "description": (
            "Configuration for FreeRTOS hooks"
        ),
        "example": "see example config",
        "group": "FREERTOS",
    })

    filename_freertos_header: str = field(metadata={
        "description": (
            "Filename of the header file with FreeRTOS configuration.",
            "Only used if the 'freertos_enable' field is true."
        ),
        "example": '"littletrace_freertos_header.h"',
        "group": "FREERTOS",
    })

    freertos_queues: list = field(metadata={
        "description": (
            "List of all FreeRTOS queue names. ",
            "Only used if the 'freertos_enable' field is true.",
        ),
        "example": '[ "ADC", "EVENTS", "UART" ]',
        "group": "FREERTOS",
    })

    freertos_tasks: list = field(metadata={
        "description": (
            "List of all FreeRTOS task names. ",
            "Only used if the 'freertos_enable' field is true.",
        ),
        "example": '[ "taskADC", "taskEVENTS", "taskUART" ]',
        "group": "FREERTOS",
    })

    freertos_timers: list = field(metadata={
        "description": (
            "List of all FreeRTOS timer names. ",
            "Only used if the 'freertos_enable' field is true.",
        ),
        "example": '[ "timerADC", "timerEVENTS", "timerUART" ]',
        "group": "FREERTOS",
    })

    freertos_event_id_prefix: list = field(metadata={
        "description": (
            "A prefix to be prepended to the names of all FreeRTOS trace evend IDs",
            "Only used if the 'freertos_enable' field is true.",
        ),
        "example": '"TRACE_EVENT_FREERTOS_"',
        "group": "FREERTOS",
    })

    freertos_hash_arg_type: list = field(metadata={
        "description": (
            "Hash type"
        ),
        "example": '"uint8_t"',
        "group": "FREERTOS",
    })

    def entry_size(self) -> int:
        n_bytes = n_bytes_for_datatype(self.typename_event_id)
        if self.enable_timestamp:
            n_bytes += n_bytes_for_datatype(self.typename_timestamp)
        for datatype in self.data_members.values():
            n_bytes += n_bytes_for_datatype(datatype)
        return n_bytes

    def offset_event_id(self):
        if self.enable_timestamp:
            return n_bytes_for_datatype(self.typename_timestamp)
        else:
            return 0

    def offset_of(self, member):
        n_bytes = n_bytes_for_datatype(self.typename_event_id)
        if self.enable_timestamp:
            n_bytes += n_bytes_for_datatype(self.typename_timestamp)
        for candidate, datatype in self.data_members.items():
            if member == candidate:
                return n_bytes
            n_bytes += n_bytes_for_datatype(datatype)
        raise KeyError(member)


    def counter_padding_bytes(self) -> int:
        n_bytes_counter = n_bytes_for_datatype(self.typename_counter)
        n_bytes_entry = self.entry_size()
        if n_bytes_entry >= n_bytes_counter:
            return n_bytes_entry - n_bytes_counter
        else:
            raise ValueError(n_bytes_counter, n_bytes_entry)

    def data_member_index(self, name):
        for i, member in enumerate(self.data_members):
            if member == name:
                return 1 + i
        raise ValueError(name)

    def trace_update_n_args(self):
        return 1 + len(self.data_members)

    def names(self):
        return self.freertos_queues + self.freertos_tasks + self.freertos_timers

    def name_hash_fixed_substring_sum(self, name, start, stop):
        return sum(map(ord, name[start:stop])) % 256

    def name_hash_string_sum(self, name):
        return sum(map(ord, name)) % (256 ** n_bytes_for_datatype(self.data_members[self.freertos_name_arg]))

    def name_hash_table(self):
        spec = self.get_hash_function_spec()
        table = { 0: "???" }
        if spec["strategy"] == "fixed_substring_sum":
            for name in self.names():
                table[self.name_hash_fixed_substring_sum(name, spec["start"], spec["stop"])] = name
        elif spec["strategy"] == "string_sum":
            for name in self.names():
                table[self.name_hash_string_sum(name)] = name
        elif spec["strategy"] == "strlen":
            for name in self.names():
                table[len(name)] = name
        else:
            raise ValueError(spec["strategy"])
        table[0] = "(unknown)"
        return table

    def get_hash_params_fixed_substring_sum(self):
        # We are allowed to use the null character in the end for the hash
        names = [name + '\0' for name in self.names()]

        # Only access characters that we know will be valid for all names
        min_len = min(len(name) for name in names)

        # Within the first min_len characters, check each possible substring offset and length
        # If the character sum of such a substring is unique for each name, we can use that as a hash function
        ranges: list[tuple[int, int]] = []
        for start_index in range(min_len):
            for stop_index in range(start_index + 1, min_len+1):
                hashes = [self.name_hash_fixed_substring_sum(name, start_index, stop_index) for name in names]
                if any(hashed == 0 for hashed in hashes):
                    # 0 is reserved for the null pointer
                    pass
                elif len(set(hashes)) != len(names):
                    # Not all unique
                    pass
                else:
                    ranges.append((start_index, stop_index))

        if ranges:
            start, stop = min(ranges, key=lambda range: range[1] - range[0])
            result = { "strategy": "fixed_substring_sum", "start": start, "stop": stop }
        else:
            result = None
        return result

    def get_hash_params_strlen(self):
        # We are allowed to use the null character in the end for the hash
        names = [name + '\0' for name in self.names()]

        # Only access characters that we know will be valid for all names
        if len({len(name) for name in names}) == len(names):
            result = {
                "strategy": "strlen",
                "max_len": max(len(name) for name in names)
            }
        else:
            result = None
        return result

    def get_hash_params_string_sum(self):
        # We are allowed to use the null character in the end for the hash
        names = [name + '\0' for name in self.names()]

        if len({self.name_hash_string_sum(name) for name in names}) == len(names):
            result = {
                "strategy": "string_sum",
                "max_len": max(len(name) for name in names)
            }
        else:
            result = None
        return result

    def get_hash_function_spec(self):
        # Check cache
        previous_result = getattr(self, "_hash_function_spec", None)
        if previous_result is not None:
            return previous_result

        if (params := self.get_hash_params_fixed_substring_sum()):
            result = params
        elif (params := self.get_hash_params_strlen()):
            result = params
        elif (params := self.get_hash_params_string_sum()):
            result = params
        else:
            raise ValueError(f"No hash function found for {self.names()}")
        self._hash_function_spec = result
        return result

    def parse_struct(self, struct):
        out = {}
        event_id = struct["event_id"]
        hook_names = list(self.freertos_hooks)
        if event_id <= len(hook_names):
            out["name"] = hook_names[event_id-1]
            config = self.freertos_hooks[out["name"]]["args"]
            for field, member in config.items():
                if member == "default":
                    member = self.freertos_default_args[field]
                if member:
                    out[field] = struct[member]
        else:
            out["name"] = f"Unknown (id={event_id})"
        name_table = self.name_hash_table()
        for field in out:
            if field.endswith("_NAME"):
                if out[field] in name_table:
                    out[field] = name_table[out[field]]
        out |= struct
        return out

###########################################################################################################
# Binary parser
###########################################################################################################

class BinaryParser:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.offset = 0

    def skip(self, n_bytes):
        self.offset += n_bytes

    def get(self, datatype):
        n_bytes = n_bytes_for_datatype(datatype)
        result = int.from_bytes(self.data[self.offset:self.offset+n_bytes], "little")
        self.offset += n_bytes
        return result

    def get_count(self):
        if self.config.enable_counter:
            count = self.get(self.config.typename_counter)
            self.skip(self.config.counter_padding_bytes())
        else:
            count = n_entries
        return count

    def get_next_entry(self):
        entry = {}
        if self.config.enable_timestamp:
            entry["timestamp"] = self.get(self.config.typename_timestamp)
        entry["event_id"] = self.get(self.config.typename_event_id)
        for member, datatype in self.config.data_members.items():
            entry[member] = self.get(datatype)
        return entry

    def sort_entries(self, entries, count):
        if self.config.enable_counter:
            if count <= len(entries):
                return entries
            else:
                cutoff = count % self.config.n_entries
                entries_by_counter = entries[cutoff:] + entries[:cutoff]
                if self.config.enable_timestamp:
                    entries_by_timestamp = sorted(entries, key=lambda entry: entry["timestamp"])
                    assert entries_by_timestamp == entries_by_counter
                return entries_by_counter
        elif self.config.enable_timestamp:
            entries_by_timestamp = sorted(entries, key=lambda entry: entry["timestamp"])
            return entries_by_timestamp
        else:
            print("Warning: no counter or timestamp enabled, so events may be out of order")
            return entries

    def get_all(self):
        count = self.get_count()
        entry_size = self.config.entry_size()
        event_id_n_bytes = n_bytes_for_datatype(self.config.typename_event_id)
        entries = [self.get_next_entry() for _ in range(min(self.config.n_entries, count))]
        entries = self.sort_entries(entries, count)
        return entries

###########################################################################################################
# Code generation (C and markdown)
###########################################################################################################

class CodeGenerator:
    def __init__(self, config):
        self.config = config

    def hfile_text(self) -> str:
        return self.cleanse_file(TEMPLATE_HFILE.format(
            pre_include=f'#include "{self.config.pre_include}"' if self.config.pre_include else "",
            definition_include_guard=self.config.definition_include_guard,
            typename_trace_entry=self.config.typename_trace_entry,
            typename_trace_buffer=self.config.typename_trace_buffer,
            function_prototypes=self.join_lines(self.prototypes(), indentation_level=0),
            definition_n_entries=self.config.definition_n_entries,
            fields_buffer=self.join_lines(self.buffer_struct_fields(), indentation_level=1),
            n_entries=self.config.n_entries,
            fields=self.join_lines(self.entry_struct_fields(), indentation_level=1),
            impl=self.implementation_in_header(),
            code_static_assertions=self.code_static_assertions(),
        ))

    def cfile_text(self) -> str:
        return self.cleanse_file(TEMPLATE_CFILE.format(
            includes=self.implementation_includes(),
            typename_trace_entry=self.config.typename_trace_entry,
            typename_counter=self.config.typename_counter,
            init_function_prototype=self.init_function_prototype(),
            update_function_prototype=self.update_function_prototype(),
            function_name_begin_update=self.config.function_name_begin_update,
            function_name_end_update=self.config.function_name_end_update,
            definition_n_entries=self.config.definition_n_entries,
            code_counter_update=self.code_counter_update(),
            code_timestamp_update=self.code_timestamp_update(),
            code_data_update=self.code_data_update(),
            code_index_update=self.code_index_update(),
            typename_index=self.typename_index(),
            typename_trace_buffer=self.config.typename_trace_buffer,
        ))

    def freertos_config_text(self) -> str:
        event_id_definitions = []
        trace_hook_definitions = []
        trace_hook_macros = []

        freertos_hook_config = self.config.freertos_hooks

        for event_id, (hook, config) in enumerate(freertos_hook_config.items(), start=1):
            event_id_definitions.append(f"#define TRACE_EVENT_FREERTOS_{hook} {event_id}")
            if config["define_trace_hook"]:
                trace_hook_definitions.append(f"#define trace{hook} TRACE_HOOK_{hook}")
            if config["generate_macro"]:
                trace_hook_macros.append(self.freertos_trace_call(hook, config))
        return self.cleanse_file(TEMPLATE_FREERTOS_CONFIG.format(
            header=self.config.filename_header,
            event_id_definitions=self.join_lines(event_id_definitions, indentation_level=0),
            definition_include_guard=self.config.definition_include_guard,
            trace_hook_definitions=self.join_lines(trace_hook_definitions, indentation_level=0),
            trace_hook_macros=self.join_lines(trace_hook_macros, indentation_level=0),
            hash_function=self.get_hash_function(),
            indentation_level=0
        ))

    def testfile_text(self) -> str:
        return self.cleanse_file(TEMPLATE_TESTFILE.format(
            code_include_header=self.code_include_header_from_test(),
            freertos_header=self.code_freertos_test_mocks(),
            function_name_test_main=self.config.function_name_test_main,
            function_name_init_trace=self.config.function_name_init_trace,
            typename_trace_entry=self.config.typename_trace_entry,
            typename_trace_buffer=self.config.typename_trace_buffer,
            size_trace_entry=self.config.entry_size(),
            definition_n_entries=self.config.definition_n_entries,
            code_initialize_timestamp=self.code_initialize_timestamp(),
            code_assert_entry_is_zero=self.code_assert_entry_is_zero(),
            code_generate_test_values=self.code_generate_test_values(),
            code_call_trace_update=self.code_call_trace_update(),
            code_check_struct_values=self.code_check_struct_values(),
            code_check_byte_values=self.code_check_byte_values(),
            function_name_begin_update=self.config.function_name_begin_update,
            function_name_end_update=self.config.function_name_end_update,
            function_name_get_timestamp=self.config.function_name_get_timestamp,
            typename_timestamp=self.config.typename_timestamp,
            code_freertos_trace_calls=self.code_freertos_trace_calls(),
        ))

    def config_documentation(self) -> str:
        lines = ["## Configuration fields", ""]
        for group_name, description in GROUP_DOCS.items():
            lines.extend([f"### {group_name}" "", description, ""])
            for field in fields(self.config):
                if field.metadata["group"] == group_name:
                    lines.extend([
                        f"#### {field.name}: {field.type.__name__}", "",
                        "\n".join(field.metadata['description']), "",
                        "Example:", "",
                        f'    "{field.name}": {field.metadata["example"]}', "",
                    ])
        return "\n".join(lines)

    def join_lines(self, lines: list, *, indentation_level: int) -> str:
        return ("\n" + "    " * indentation_level).join(lines)

    def cleanse_file(self, text):
        lines = text.strip().split("\n")
        out = []
        last_was_empty = False
        for line in lines:
            line = line.rstrip()
            if line or not last_was_empty:
                out.append(line)
            last_was_empty = not line
        out.append("")
        return "\n".join(out)

    def entry_struct_fields(self) -> str:
        fields = []
        if self.config.enable_timestamp:
            fields.append(f"{self.config.typename_timestamp} timestamp;")
        fields.append(f"{self.config.typename_event_id} event_id;")
        for name, datatype in self.config.data_members.items():
            fields.append(f"{datatype} {name};")
        return fields

    def buffer_struct_fields(self) -> str:
        fields_buffer = []
        if self.config.enable_counter:
            counter_struct_fields = [f"{self.config.typename_counter} counter;"]
            n_padding_bytes_counter = self.config.counter_padding_bytes()
            if n_padding_bytes_counter:
                counter_struct_fields.append(f"uint8_t _padding[{n_padding_bytes_counter}];")
            counter_struct_fields = " ".join(counter_struct_fields)
            fields_buffer.append(f"struct __attribute__((__packed__)) {{ {counter_struct_fields} }} padded_counter;")
        fields_buffer.append(f"{self.config.typename_trace_entry} entries[{self.config.definition_n_entries}];")
        return fields_buffer

    def prototypes(self):
        prototypes = []
        if self.config.enable_timestamp:
            prototypes.append(f"{self.config.typename_timestamp} {self.config.function_name_get_timestamp}(void);")
        prototypes.append(f"{self.init_function_prototype()};")
        prototypes.append(f"{self.update_function_prototype()};")
        return prototypes

    def implementation_in_header(self):
        if self.config.header_only:
            return self.join_lines([
                f"#if {self.config.definition_implementation_guard}",
                self.cfile_text(),
                "#endif"
            ], indentation_level=0)
        else:
            return ""

    def init_function_prototype(self) -> str:
        return f"void {self.config.function_name_init_trace}({self.config.typename_trace_buffer} *buffer)"

    def update_function_prototype(self) -> str:
        update_args = [f"{self.config.typename_event_id} event_id"] + [f"{datatype} {name}" for name, datatype in self.config.data_members.items()]
        return f"void {self.config.function_name_update_trace}({', '.join(update_args)})"

    def get_function_prototypes_text(self) -> str:
        return self.join_lines(prototypes, indentation_level=0)

    def implementation_includes(self) -> str:
        include_filenames = []
        if not self.config.header_only:
            include_filenames.append(f'"{self.hfile}"')
        include_filenames.append("<string.h>")
        return self.join_lines((f"#include {filename}" for filename in include_filenames), indentation_level=1)

    def code_counter_update(self):
        if self.config.enable_counter:
            return f"g_trace_state.buffer->padded_counter.counter++;"
        else:
            return ""

    def code_timestamp_update(self):
        if self.config.enable_timestamp:
            return f"entry->timestamp = {self.config.function_name_get_timestamp}();"
        else:
            return ""

    def code_data_update(self):
        return self.join_lines((
            f"entry->{name} = {'*' if '*' in datatype else ''}{name};"
            for name, datatype in self.config.data_members.items()
        ), indentation_level=1)

    def typename_index(self):
        if self.config.n_entries == 256:
            return "uint8_t"
        elif self.config.n_entries == 65536:
            return "uint16_t"
        else:
            return "size_t"

    def code_index_update(self):
        if self.typename_index() == "size_t":
            is_power_of_two = bin(self.config.n_entries).count("1") == 1
            if is_power_of_two:
                return f"g_trace_state.index = (g_trace_state.index + 1) & {self.config.n_entries-1:#x};"
            else:
                return self.join_lines((
                    f"size_t next_index = g_trace_state.index + 1;",
                    f"g_trace_state.index = next_index == {self.config.definition_n_entries} ? 0 : next_index;"
                ), indentation_level=1)
        else:
            return "g_trace_state.index++"

    def code_include_header_from_test(self):
        if self.config.header_only:
            return self.join_lines([
                f"#define {self.config.definition_implementation_guard} 1",
                f'#include "{self.config.filename_header}"',
                f"#undef {self.config.definition_implementation_guard}"
            ], indentation_level=0)
        else:
            return f'#include "{self.config.filename_header}"'

    def offset_assertions(self, struct, member, datatype, offset):
        return [
            f'static_assert(offsetof({struct}, {member}) == {offset}, "{member} does not have the expected offset");',
            f'static_assert(offsetof({struct}, {member}) % sizeof({datatype}) == 0, "{member} is misaligned");',
        ]

    def code_static_assertions(self):
        assertions = []
        assertions.extend(self.offset_assertions(self.config.typename_trace_entry, "event_id", self.config.typename_event_id, self.config.offset_event_id()))
        for member, datatype in self.config.data_members.items():
            assertions.extend(self.offset_assertions(self.config.typename_trace_entry, member, datatype, self.config.offset_of(member)))
        if self.config.enable_counter:
            assertions.append(f'static_assert(sizeof((({self.config.typename_trace_buffer}*)(0))->entries[0]) == sizeof((({self.config.typename_trace_buffer}*)(0))->padded_counter), "Counter should occupy the same space as one entry");')
        assertions.append(f'static_assert(sizeof({self.config.typename_trace_entry}) == {self.config.entry_size()}, "{self.config.typename_trace_entry} does not have the expected size");')
        assertions.append(f'static_assert(sizeof({self.config.typename_trace_buffer}) == {(self.config.n_entries + self.config.enable_counter) * self.config.entry_size()}, "{self.config.typename_trace_buffer} does not have the expected size");')
        assertions.extend(self.offset_assertions(self.config.typename_trace_buffer, "entries", self.config.typename_trace_entry, self.config.enable_counter * self.config.entry_size()))
        return self.join_lines(assertions, indentation_level=0)

    def code_assert_entry_is_zero(self):
        assertions = []
        if self.config.enable_timestamp:
            assertions.append(f"{self.config.function_name_assert}(entry->timestamp == 0);")
        assertions.append(f"{self.config.function_name_assert}(entry->event_id == 0);")
        for name in self.config.data_members:
            assertions.append(f"{self.config.function_name_assert}(entry->{name} == 0);")
        return self.join_lines(assertions, indentation_level=2)

    def code_generate_test_values(self):
        values = []
        values.append(f"{self.config.typename_event_id} value_event_id = ({self.config.typename_event_id})(i+1);")
        for j, (name, datatype) in enumerate(self.config.data_members.items(), start=2):
            values.append(f"{datatype} value_{name} = ({datatype})(i+{j});")
        return self.join_lines(values, indentation_level=2)

    def code_call_trace_update(self):
        args = []
        args.append("value_event_id")
        args.extend(f"value_{name}" for name in self.config.data_members)
        return f"{self.config.function_name_update_trace}({', '.join(args)});"

    def code_check_struct_values(self):
        assertions = []
        if self.config.enable_timestamp:
            assertions.append(f"value_timestamp++;")
            assertions.append(f"assert(entry->timestamp == value_timestamp);")
        assertions.append(f"assert(entry->event_id == value_event_id);")
        for name in self.config.data_members:
            assertions.append(f"assert(entry->{name} == value_{name});")
        return self.join_lines(assertions, indentation_level=2)

    def code_check_byte_values(self):
        assertions = []
        if self.config.enable_timestamp:
            assertions.extend((
                f"assert(*({self.config.typename_timestamp}*)&byte_data[byte_index] == value_timestamp);",
                f"byte_index += sizeof(value_timestamp);"
            ))
        assertions.extend((
            f"assert(*({self.config.typename_event_id}*)&byte_data[byte_index] == value_event_id);",
            f"byte_index += sizeof(value_event_id);"
        ))
        for name, datatype in self.config.data_members.items():
            assertions.extend((
                f"assert(*({datatype}*)&byte_data[byte_index] == value_{name});",
                f"byte_index += sizeof(value_{name});"
            ))
        return self.join_lines(assertions, indentation_level=2)

    def code_initialize_timestamp(self):
        if self.config.enable_timestamp:
            result = f"{self.config.typename_timestamp} value_timestamp = 0;"
        else:
            result = ""
        return result

    def code_freertos_test_mocks(self):
        if not self.config.freertos_enable:
            return ""

        return TEMPLATE_CODE_FREERTOS_TEST_MOCKS.format(
            filename_freertos_header=self.config.filename_freertos_header,
            task_names=(", ".join(f'"{name}"' for name in self.config.freertos_tasks)),
            queue_names=(", ".join(f'"{name}"' for name in self.config.freertos_queues)),
            timer_names=(", ".join(f'"{name}"' for name in self.config.freertos_timers)),
        )

    def freertos_trace_call(self, name, config):
        macro_name = f"TRACE_HOOK_{name}"
        macro_args = []
        function_args = [f"{self.config.freertos_event_id_prefix}{name}"] + ["0" for _ in range(self.config.trace_update_n_args()-1)]
        macro_code = []

        if sum(name in hooks for hooks in FREERTOS_TRACE_HOOKS.values()) != 1:
            raise ValueError(f"Unknown hook: {name}")

        for arg_types, hooks in FREERTOS_TRACE_HOOKS.items():
            if name in hooks:
                args = (name, self.config, function_args, config)
                if "QUEUE_HANDLE" in arg_types:
                    maybe_add_queue_length(*args)
                    maybe_add_queue_name(*args)
                    maybe_add_queue_pointer(*args)
                if "TASK_HANDLE" in arg_types:
                    maybe_add_task_name(*args)
                    maybe_add_task_pointer(*args)
                if "PRIORITY" in arg_types:
                    maybe_add_task_priority(*args)
                if "QUEUE_TYPE" in arg_types:
                    maybe_add_queue_type(*args)
                if "TICK_COUNT" in arg_types:
                    maybe_add_tick_count(*args)
                if "TIMER" in arg_types:
                    maybe_add_timer_pointer(*args)
                    if "RETURN" in arg_types:
                        maybe_add_timer_return_value(*args)
                    if "COMMAND_ID" in arg_types:
                        maybe_add_timer_command_id(*args)
                    if "COMMAND_VALUE" in arg_types:
                        maybe_add_timer_command_value(*args)
                if "STREAM_BUFFER" in arg_types:
                    maybe_add_stream_buffer_pointer(*args)
                if "IS_MESSAGE_BUFFER" in arg_types:
                    maybe_add_is_message_buffer(*args)
                if "BYTE_COUNT" in arg_types:
                    maybe_add_byte_count(*args)
                if "EVENT_GROUP" in arg_types:
                    if "BITS" in arg_types:
                        maybe_add_bits(*args)
                    if "BITS_2" in arg_types:
                        maybe_add_bits_2(*args)
                    if "TIMEOUT_OCCURRED" in arg_types:
                        maybe_add_timeout_occurred(*args)
                if "ADDRESS" in arg_types:
                    maybe_add_address(*args)
                    if "PARAM_1" in arg_types:
                        maybe_add_param_1(*args)
                    if "PARAM_2" in arg_types:
                        maybe_add_param_2(*args)
                    if "RETURN" in arg_types:
                        maybe_add_callback_returns(*args)
                if "QUEUE_NAME" in arg_types:
                    maybe_add_queue_name(*args)
                if "INDEX" in arg_types:
                    maybe_add_index(*args)

                # Special cases
                if name in { "TASK_SWITCHED_IN", "TASK_SWITCHED_OUT" }:
                    maybe_add_current_task_name(*args)
                    maybe_add_current_task_pointer(*args)

                macro_args.extend(arg_types)
                break

        macro_arglist = ', '.join(macro_args)
        function_args = [
            f"    {arg} "
            if i == len(function_args)-1
            else f"    {arg},"
            for i, arg
            in enumerate(function_args)
        ]

        macro_code.extend([f"{self.config.function_name_update_trace}(", *function_args, ");"])
        max_macro_line_len = max(len(line) for line in macro_code)
        macro_code = [line + " " * (max_macro_line_len - len(line) + 1) for line in macro_code]
        macro_code = "\\\n    ".join(macro_code)

        return f"#define {macro_name}({macro_arglist}) do{{ \\\n    {macro_code}\\\n}} while (0)\n"

    def code_freertos_trace_calls(self):
        if not self.config.freertos_enable:
            return ""

        calls = []
        for arg_types, hooks in FREERTOS_TRACE_HOOKS.items():
            for hook in hooks:
                args = ', '.join(str(i) for i in range(len(arg_types)))
                calls.append(f"trace{hook}({args});")
        return self.join_lines(calls, indentation_level=1)

    def get_hash_function_fixed_substring_sum(self, spec):
        if spec["strategy"] != "fixed_substring_sum":
            return None
        hash_doc_lines = [
            f"// Uses the sum of some characters in the name as the hash",
            f"// Known names and corresponding hashes: ",
        ]
        for hash, name in self.config.name_hash_table().items():
            if name == "(unknown)":
                hash_doc_lines.append("//     Unknown => 0")
            else:
                name_terminated = name + "\0"
                hash_calc_chars = " + ".join(f"'{name_terminated[i]}'" if name_terminated[i] != '\0' else "'\\0'" for i in range(spec["start"], spec["stop"]))
                hash_calc_ords = " + ".join(f"{ord(name_terminated[i])}" for i in range(spec["start"], spec["stop"]))
                hash_doc_lines.append(f'//     "{name}" => {hash_calc_chars} = {hash_calc_ords} = {hash}')
        hash_function = TEMPLATE_HASH_FUNCTION_FIXED_SUBSTRING.format(
            hash_doc=self.join_lines(hash_doc_lines, indentation_level=0),
            datatype=self.config.freertos_hash_arg_type,
            substring_sum=" + ".join([f"name_u8[{i}]" for i in range(spec["start"], spec["stop"])])
        )
        return hash_function

    def get_hash_function_strlen(self, spec):
        if spec["strategy"] != "strlen":
            return None
        hash_doc_lines = [
            f"// Uses the string length of the name as the hash",
            f"// Known names and corresponding hashes:"
        ]
        for hash, name in self.config.name_hash_table().items():
            if name == "(unknown)":
                hash_doc_lines.append("//     Unknown => 0")
            else:
                hash_doc_lines.append(f'//     "{name}" => {hash}')
        hash_function = TEMPLATE_HASH_FUNCTION_STRLEN.format(
            hash_doc=self.join_lines(hash_doc_lines, indentation_level=0),
            datatype=self.config.freertos_hash_arg_type,
            max_len=spec["max_len"],
        )
        return hash_function

    def get_hash_function_string_sum(self, spec):
        if spec["strategy"] != "string_sum":
            return None
        hash_doc_lines = [
            f"// Uses the sum of the characters in the name as the hash",
            f"// Known names and corresponding hashes: ",
        ]
        for hash, name in self.config.name_hash_table().items():
            if name == "(unknown)":
                hash_doc_lines.append("//     Unknown => 0")
            else:
                hash_calc_chars = " + ".join(f"'{c}'" for c in name)
                hash_calc_ords = " + ".join(f"{ord(c)}" for c in name)
                hash_doc_lines.append(f'//     "{name}" => {hash_calc_chars} = {hash_calc_ords} = {hash}')
        hash_function = TEMPLATE_HASH_FUNCTION_STRSUM.format(
            hash_doc=self.join_lines(hash_doc_lines, indentation_level=0),
            datatype=self.config.freertos_hash_arg_type,
            max_len=spec["max_len"]
        )

        return hash_function

    def get_hash_function(self):
        if self.config.freertos_hash_arg_type:
            spec = self.config.get_hash_function_spec()
            if (result := self.get_hash_function_fixed_substring_sum(spec)):
                return result
            elif (result := self.get_hash_function_strlen(spec)):
                return result
            elif (result := self.get_hash_function_string_sum(spec)):
                return result
            else:
                raise ValueError(spec)
        else:
            return ""
        return hash_function


###########################################################################################################
# Helpers for generating trace calls
###########################################################################################################

def freertos_trace_call_helper(macro_name):
    def decorator(func):
        def wrapper(hook_name, main_config, function_args, config):
            member_name = config["args"][macro_name]
            if member_name == "default":
                member_name = main_config.freertos_default_args[macro_name]
            if member_name:
                try:
                    datatype = main_config.data_members[member_name]
                except KeyError:
                    raise KeyError(f"While generating trace macro for {hook_name}: no data member called '{member_name}', available members are {list(main_config.data_members)}") from None
                index = main_config.data_member_index(member_name)
                value = func(datatype)
                if function_args[index] != "0":
                    print(f"Note: for FreeRTOS hook {hook_name}, config wants to set {member_name} to both \"{function_args[index]}\" and \"{value}\". The latter will be used.")
                function_args[index] = value
        return wrapper
    return decorator

def int_converter(name):
    return freertos_trace_call_helper(name)(lambda datatype: f"({datatype})({name})")

def pointer_converter(config_name, macro_arg):
    return freertos_trace_call_helper(config_name)(lambda datatype: pointer_cast_code(datatype, macro_arg))

@freertos_trace_call_helper("QUEUE_LENGTH")
def maybe_add_queue_length(datatype):
    return f"({datatype})uxQueueMessagesWaiting(QUEUE_HANDLE)"

@freertos_trace_call_helper("QUEUE_NAME")
def maybe_add_queue_name(datatype):
    return f"_get_name_hash(pcQueueGetName(QUEUE_HANDLE))"

@freertos_trace_call_helper("TASK_NAME")
def maybe_add_task_name(datatype):
    return f"_get_name_hash(pcTaskGetName(TASK_HANDLE))"

@freertos_trace_call_helper("TASK_NAME")
def maybe_add_current_task_name(datatype):
    return f"_get_name_hash(pcTaskGetName(pxCurrentTCB))"

@freertos_trace_call_helper("TIMER_NAME")
def maybe_add_timer_name(datatype):
    return f"_get_name_hash(pcTimerGetName(TIMER))"

maybe_add_task_priority         = int_converter("PRIORITY")
maybe_add_queue_type            = int_converter("QUEUE_TYPE")
maybe_add_tick_count            = int_converter("TICK_COUNT")
maybe_add_timer_command_id      = int_converter("COMMAND_ID")
maybe_add_timer_command_value   = int_converter("COMMAND_VALUE")
maybe_add_timer_return_value    = int_converter("RETURN")
maybe_add_is_message_buffer     = int_converter("IS_MESSAGE_BUFFER")
maybe_add_byte_count            = int_converter("BYTE_COUNT")
maybe_add_bits                  = int_converter("BITS")
maybe_add_bits_2                = int_converter("BITS_2")
maybe_add_timeout_occurred      = int_converter("TIMEOUT_OCCURRED")
maybe_add_param_1               = int_converter("PARAM_1")
maybe_add_param_2               = int_converter("PARAM_2")
maybe_add_callback_returns      = int_converter("RETURN")
maybe_add_index                 = int_converter("INDEX")
maybe_add_timer_pointer         = pointer_converter("TIMER_POINTER", "TIMER")
maybe_add_task_pointer          = pointer_converter("TASK_POINTER", "TASK_HANDLE")
maybe_add_queue_pointer         = pointer_converter("QUEUE_POINTER", "QUEUE_HANDLE")
maybe_add_stream_buffer_pointer = pointer_converter("STREAM_BUFFER_POINTER", "STREAM_BUFFER")
maybe_add_address               = pointer_converter("ADDRESS", "ADDRESS")
maybe_add_current_task_pointer  = pointer_converter("TASK_POINTER", "pxCurrentTCB")


###########################################################################################################
# Helpers for datatypes
###########################################################################################################

def n_bytes_for_datatype(datatype):
    if "int8_t" in datatype:
        return 1
    elif "int16_t" in datatype:
        return 2
    elif "int32_t" in datatype:
        return 4
    elif "int64_t" in datatype:
        return 8
    else:
        raise ValueError(datatype)

def is_unsigned(datatype):
    return "uint" in datatype

def and_value(datatype):
    return hex((256 ** n_bytes_for_datatype(datatype)) - 1)

def pointer_cast_code(datatype, value):
    intptr_type = "uintptr_t" if is_unsigned(datatype) else "intptr_t"
    return f"({datatype})(({intptr_type})({value}) & {and_value(datatype)})"

###########################################################################################################
# Constants
###########################################################################################################

GROUP_DOCS = {
    "DATA": "These fields affect the layout and contents of the trace buffer. ",
    "API": "These fields affect how the library is integrated into other code and include mostly names. ",
    "HOOKS": "These are snippets of code which can help customize the behaviour of the library. ",
    "TESTING": "These fields only affect the generated test code. ",
    "STYLE": "These fields affect the style of the generated code. ",
    "FREERTOS": "These fields affect integration with FreeRTOS. ",
}

TEMPLATE_HFILE = """
#ifndef {definition_include_guard}
#define {definition_include_guard}

#ifdef __cplusplus
    extern "C" {{
#endif

{pre_include}

#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define {definition_n_entries} {n_entries}

typedef struct  __attribute__((__packed__)) {typename_trace_entry}
{{
    {fields}
}} {typename_trace_entry};

typedef struct __attribute__((__packed__)) {typename_trace_buffer}
{{
    {fields_buffer}
}} {typename_trace_buffer};

{code_static_assertions}

{function_prototypes}

#ifdef __cplusplus
    }}
#endif

#endif

{impl}

"""

TEMPLATE_CFILE = """
{includes}

static volatile struct trace_state_t
{{
    {typename_trace_buffer} *buffer;
    {typename_index} index;
}} g_trace_state;

{init_function_prototype}
{{
    g_trace_state.buffer = buffer;
    g_trace_state.index = 0;
    memset(buffer, 0, sizeof({typename_trace_buffer}));
}}

{update_function_prototype}
{{
    {function_name_begin_update}();
    {typename_trace_entry} *entry = &g_trace_state.buffer->entries[g_trace_state.index];
    {code_timestamp_update}
    entry->event_id = event_id;
    {code_data_update}
    {code_index_update}
    {code_counter_update}
    {function_name_end_update}();
}}
"""

TEMPLATE_TESTFILE = """
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

{freertos_header}

void {function_name_begin_update}(void)
{{
}}

void {function_name_end_update}(void)
{{
}}

{typename_timestamp} {function_name_get_timestamp}(void)
{{
    static {typename_timestamp} timestamp = 0;
    timestamp++;
    return timestamp;
}}

{code_include_header}

int {function_name_test_main}(void)
{{
    static {typename_trace_buffer} trace_buffer = {{}};

    memset(&trace_buffer, 0xff, sizeof(trace_buffer));
    {function_name_init_trace}(&trace_buffer);

    for (size_t i = 0; i < {definition_n_entries}; i++)
    {{
        {typename_trace_entry} *entry = &trace_buffer.entries[i];
        {code_assert_entry_is_zero}
    }}

    {code_initialize_timestamp}
    for (size_t i = 0; i < {definition_n_entries}*5; i++)
    {{
        {code_generate_test_values}

        {code_call_trace_update}

        size_t entry_index = i % {definition_n_entries};
        {typename_trace_entry} *entry = &trace_buffer.entries[entry_index];
        {code_check_struct_values}

        uint8_t *byte_data = (uint8_t*)&trace_buffer.entries[0];
        size_t byte_index = entry_index * sizeof({typename_trace_entry});
        {code_check_byte_values}
    }}

    {code_freertos_trace_calls}

    FILE *fh = fopen("trace_data.bin", "wb");
    if (fh != NULL)
    {{
        fwrite(&trace_buffer, sizeof(trace_buffer), 1, fh);
        fclose(fh);
    }}

    printf("All tests passed.\\n");
    return 0;
}}
"""

TEMPLATE_CODE_FREERTOS_TEST_MOCKS = """
#include "{filename_freertos_header}"

// Stub implementation of what we need from FreeRTOS"
static const char* task_names[] = {{ {task_names} }};
static const char* queue_names[] = {{ {queue_names} }};
static const char* timer_names[] = {{ {timer_names} }};

const char* pcTaskGetName(void *arg)
{{
    static size_t index = 0;
    const char *out = task_names[index % sizeof(task_names)/sizeof(task_names[0])];
    index++;
    return out;
}}

const char* pcQueueGetName(void *arg)
{{
    static size_t index = 0;
    const char *out = queue_names[index % sizeof(queue_names)/sizeof(queue_names[0])];
    index++;
    return out;
}}

const char* pcTimerGetName(void *arg)
{{
    static size_t index = 0;
    const char *out = timer_names[index % sizeof(timer_names)/sizeof(timer_names[0])];
    index++;
    return out;
}}

int uxQueueMessagesWaiting(void *arg)
{{
    static int result = 0;
    result++;
    return result;
}}

struct tskTaskControlBlock
{{
    int x;
}} xCurrentTCB;

struct tskTaskControlBlock* volatile pxCurrentTCB = &xCurrentTCB;
"""

TEMPLATE_FREERTOS_CONFIG = """
#ifndef {definition_include_guard}__FREERTOS_CONFIG_
#define {definition_include_guard}__FREERTOS_CONFIG_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "{header}"

// Some trace hooks don't pass the current task, but we know it's going to be the same as pxCurrentTCB
extern struct tskTaskControlBlock* volatile pxCurrentTCB;

// All event IDs
{event_id_definitions}

// Define trace hooks for all hooks configured with define_trace_hook=true
// This will make them recognized by FreeRTOS
{trace_hook_definitions}

// Define macros for each trace hook
{trace_hook_macros}

// Hash function for names
{hash_function}

#endif
"""

TEMPLATE_HASH_FUNCTION_FIXED_SUBSTRING = """
{hash_doc}
static inline {datatype} _get_name_hash(const char *name)
{{
    if (name == NULL)
    {{
        return 0;
    }}
    else
    {{
        // Cast to uint8_t to ensure proper overflow semantics
        const uint8_t* name_u8 = (uint8_t*)name;
        return ({datatype})({substring_sum});
    }}
}}
"""

TEMPLATE_HASH_FUNCTION_STRLEN = """
{hash_doc}
static inline {datatype} _get_name_hash(const char *name)
{{
    if (name == NULL)
    {{
        return 0;
    }}
    else
    {{
        for (size_t i = 0; i < {max_len}; i++)
        {{
            if (name[i] == 0)
            {{
                return i;
            }}
        }}
        return 0;
    }}
}}
"""

TEMPLATE_HASH_FUNCTION_STRSUM = """
{hash_doc}
static inline {datatype} _get_name_hash(const char *name)
{{
    if (name == NULL)
    {{
        return 0;
    }}
    else
    {{
        // Cast to uint8_t to ensure proper overflow semantics
        const uint8_t* name_u8 = (uint8_t*)name;
        {datatype} sum = 0;

        for (size_t i = 0; i < {max_len}; i++)
        {{
            if (name_u8[i] == 0)
            {{
                break;
            }}
            sum += name_u8[i];
        }}
        return sum;
    }}
}}
"""

FREERTOS_TRACE_HOOKS = {
    tuple(): {
        "CREATE_COUNTING_SEMAPHORE",
        "CREATE_COUNTING_SEMAPHORE_FAILED",
        "CREATE_MUTEX_FAILED",
        "TASK_CREATE_FAILED",
        "TASK_DELAY",
        "TASK_SWITCHED_IN",
        "TASK_SWITCHED_OUT",
        "TIMER_CREATE_FAILED",
        "EVENT_GROUP_CREATE_FAILED",
    },
    ("QUEUE_HANDLE", ): {
        "BLOCKING_ON_QUEUE_RECEIVE",
        "BLOCKING_ON_QUEUE_SEND",
        "BLOCKING_ON_QUEUE_PEEK",
        "CREATE_MUTEX",
        "GIVE_MUTEX_RECURSIVE",
        "GIVE_MUTEX_RECURSIVE_FAILED",
        "QUEUE_CREATE",
        "QUEUE_PEEK",
        "QUEUE_PEEK_FAILED",
        "QUEUE_PEEK_FROM_ISR",
        "QUEUE_PEEK_FROM_ISR_FAILED",
        "QUEUE_RECEIVE_FAILED",
        "QUEUE_RECEIVE_FROM_ISR_FAILED",
        "QUEUE_RECEIVE",
        "QUEUE_RECEIVE_FROM_ISR",
        "QUEUE_SEND",
        "QUEUE_SEND_FROM_ISR",
        "QUEUE_SEND_FAILED",
        "QUEUE_SEND_FROM_ISR_FAILED",
        "QUEUE_SET_SEND",
        "QUEUE_DELETE",
        "TAKE_MUTEX_RECURSIVE",
        "TAKE_MUTEX_RECURSIVE_FAILED",
    },
    ("QUEUE_TYPE", ): {
        "QUEUE_CREATE_FAILED"
    },
    ("TASK_HANDLE", ): {
        "MOVED_TASK_TO_READY_STATE",
        "POST_MOVED_TASK_TO_READY_STATE",
        "TASK_CREATE",
        "TASK_DELETE",
        "TASK_SUSPEND",
        "TASK_RESUME",
        "TASK_RESUME_FROM_ISR",
    },
    ("TASK_HANDLE", "PRIORITY", ): {
        "TASK_PRIORITY_SET",
    },
    ("TICK_COUNT", ): {
        "TASK_DELAY_UNTIL",
        "TASK_INCREMENT_TICK"
    },
    ("TIMER", ): {
        "TIMER_CREATE",
        "TIMER_EXPIRED",
    },
    ("TIMER", "COMMAND_ID", "COMMAND_VALUE"): {
        "TIMER_COMMAND_RECEIVED",
    },
    ("TIMER", "COMMAND_ID", "COMMAND_VALUE", "RETURN"): {
        "TIMER_COMMAND_SEND",
    },
    ("ADDRESS", "BYTE_COUNT"): {
        "MALLOC",
    },
    ("ADDRESS", ): {
        "FREE",
    },
    ("EVENT_GROUP", ): {
        "EVENT_GROUP_CREATE",
        "EVENT_GROUP_DELETE",
    },
    ("EVENT_GROUP", "BITS"): {
        "EVENT_GROUP_WAIT_BITS_BLOCK",
        "EVENT_GROUP_CLEAR_BITS",
        "EVENT_GROUP_CLEAR_BITS_FROM_ISR",
        "EVENT_GROUP_SET_BITS",
        "EVENT_GROUP_SET_BITS_FROM_ISR",
    },
    ("EVENT_GROUP", "BITS", "TIMEOUT_OCCURRED"): {
        "EVENT_GROUP_WAIT_BITS_END",
    },
    ("EVENT_GROUP", "BITS", "BITS_2"): {
        "EVENT_GROUP_SYNC_BLOCK",
    },
    ("EVENT_GROUP", "BITS", "BITS_2", "TIMEOUT_OCCURRED"): {
        "EVENT_GROUP_SYNC_END",
    },
    ("CALLBACK", "PARAM_1", "PARAM_2", "RETURN"): {
        "PEND_FUNC_CALL",
        "PEND_FUNC_CALL_FROM_ISR",
    },
    ("QUEUE_HANDLE", "STRING"): {
        "QUEUE_REGISTRY_ADD",
    },
    ("INDEX", ): {
        "TASK_NOTIFY_TAKE_BLOCK",
        "TASK_NOTIFY_TAKE",
        "TASK_NOTIFY_WAIT_BLOCK",
        "TASK_NOTIFY_WAIT",
        "TASK_NOTIFY",
        "TASK_NOTIFY_FROM_ISR",
        "TASK_NOTIFY_GIVE_FROM_ISR",
    },
    ("IS_MESSAGE_BUFFER", ): {
        "STREAM_BUFFER_CREATE_FAILED",
    },
    ("RETURN", "IS_MESSAGE_BUFFER"): {
        "STREAM_BUFFER_CREATE_STATIC_FAILED",
    },
    ("STREAM_BUFFER",): {
        "STREAM_BUFFER_DELETE",
        "STREAM_BUFFER_RESET",
        "BLOCKING_ON_STREAM_BUFFER_SEND",
        "STREAM_BUFFER_SEND_FAILED",
        "BLOCKING_ON_STREAM_BUFFER_RECEIVE",
        "STREAM_BUFFER_RECEIVE_FAILED",
    },
    ("STREAM_BUFFER", "IS_MESSAGE_BUFFER"): {
        "STREAM_BUFFER_CREATE",
    },
    ("STREAM_BUFFER", "BYTE_COUNT",): {
        "STREAM_BUFFER_SEND",
        "STREAM_BUFFER_SEND_FROM_ISR",
        "STREAM_BUFFER_RECEIVE",
        "STREAM_BUFFER_RECEIVE_FROM_ISR",
    },
}

###########################################################################################################
# Script entrypoint
###########################################################################################################

if __name__ == "__main__":
    main()
