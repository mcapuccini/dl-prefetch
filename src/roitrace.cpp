/*
 * Copyright 2002-2019 Intel Corporation.
 * 
 * This software is provided to you as Sample Source Code as defined in the accompanying
 * End User License Agreement for the Intel(R) Software Development Products ("Agreement")
 * section 1.L.
 * 
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
 */

/*
 *  This file contains an ISA-portable PIN tool for tracing memory accesses with a few mods for tracing PARSEC
 */

#include <stdio.h>
#include "pin.H"
#include <string>

const CHAR * ROI_BEGIN = "__parsec_roi_begin";
const CHAR * ROI_END = "__parsec_roi_end";

FILE * trace;
bool isROI = false;

// Print a memory read record
VOID RecordMemRead(VOID * ip, VOID * addr, CHAR * rtn)
{
    // Return if not in ROI
    if(!isROI)
    {
        return;
    }

    // Log memory access in CSV
    fprintf(trace,"%p,R,%p,%s\n", ip, addr, rtn);
}

// Print a memory write record
VOID RecordMemWrite(VOID * ip, VOID * addr, CHAR * rtn)
{
    // Return if not in ROI
    if(!isROI)
    {
        return;
    }

    // Log memory access in CSV
    fprintf(trace,"%p,W,%p,%s\n", ip, addr, rtn);
}

// Set ROI flag
VOID StartROI()
{
    isROI = true;
}

// Set ROI flag
VOID StopROI()
{
    isROI = false;
}

// Is called for every instruction and instruments reads and writes
VOID Instruction(INS ins, VOID *v)
{
    // Instruments memory accesses using a predicated call, i.e.
    // the instrumentation is called iff the instruction will actually be executed.
    //
    // On the IA-32 and Intel(R) 64 architectures conditional moves and REP 
    // prefixed instructions appear as predicated instructions in Pin.
    UINT32 memOperands = INS_MemoryOperandCount(ins);

    // Iterate over each memory operand of the instruction.
    for (UINT32 memOp = 0; memOp < memOperands; memOp++)
    {
        // Get routine name if valid
        const CHAR * name = "invalid";
        if(RTN_Valid(INS_Rtn(ins))) 
        {
            name = RTN_Name(INS_Rtn(ins)).c_str();
        }

        if (INS_MemoryOperandIsRead(ins, memOp))
        {
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead,
                IARG_INST_PTR,
                IARG_MEMORYOP_EA, memOp,
                IARG_ADDRINT, name,
                IARG_END);
        }
        // Note that in some architectures a single memory operand can be 
        // both read and written (for instance incl (%eax) on IA-32)
        // In that case we instrument it once for read and once for write.
        if (INS_MemoryOperandIsWritten(ins, memOp))
        {
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite,
                IARG_INST_PTR,
                IARG_MEMORYOP_EA, memOp,
                IARG_ADDRINT, name,
                IARG_END);
        }
    }
}

// Pin calls this function every time a new rtn is executed
VOID Routine(RTN rtn, VOID *v)
{
    // Get routine name
    const CHAR * name = RTN_Name(rtn).c_str();

    if(strcmp(name,ROI_BEGIN) == 0) {
        // Start tracing after ROI begin exec
        RTN_Open(rtn);
        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)StartROI, IARG_END);
        RTN_Close(rtn);
    } else if (strcmp(name,ROI_END) == 0) {
        // Stop tracing before ROI begin exec
        RTN_Open(rtn);
        RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)StopROI, IARG_END);
        RTN_Close(rtn);
    }
}

// Pin calls this function at the end
VOID Fini(INT32 code, VOID *v)
{
    fclose(trace);
}

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
   
INT32 Usage()
{
    PIN_ERROR( "This Pintool prints a trace of memory addresses\n" 
              + KNOB_BASE::StringKnobSummary() + "\n");
    return -1;
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char *argv[])
{
    // Initialize symbol table code, needed for rtn instrumentation
    PIN_InitSymbols();

    // Usage
    if (PIN_Init(argc, argv)) return Usage();

    // Open trace file and write header
    trace = fopen("roitrace.csv", "w");
    fprintf(trace,"pc,rw,addr,rtn\n");

    // Add instrument functions
    RTN_AddInstrumentFunction(Routine, 0);
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();
    
    return 0;
}
