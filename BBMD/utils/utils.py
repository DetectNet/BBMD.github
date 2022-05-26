from capstone import *
from capstone.x86 import *
import pefile
import os
import pickle
import numpy as np
import pandas as pd

# Disassembling Function
def get_main_code_section(sections, base_of_code):
    addresses = []
    #get addresses of all sections
    for section in sections: 
        addresses.append(section.VirtualAddress)
        
    #if the address of section corresponds to the first instruction then
    #this section should be the main code section
    if base_of_code in addresses:    
        return sections[addresses.index(base_of_code)]
    #otherwise, sort addresses and look for the interval to which the base of code
    #belongs
    else:
        addresses.append(base_of_code)
        addresses.sort()
        if addresses.index(base_of_code)!= 0:
            return sections[addresses.index(base_of_code)-1]
        else:
            #this means we failed to locate it
            return None

# Extract pefile all opcodes
def ExtractPefile12000Opcodes(opcodes, exe):
    #Opcode LIst
    opcodeList = []
    #get main code section
    main_code = get_main_code_section(exe.sections, exe.OPTIONAL_HEADER.BaseOfCode)
    #define architecutre of the machine 
    md = Cs(CS_ARCH_X86, CS_MODE_32)
    md.detail = True
    last_address = 0
    last_size = 0
    #Beginning of code section
    begin = main_code.PointerToRawData
    #the end of the first continuous bloc of code
    end = begin+main_code.SizeOfRawData
    while True:
        #parse code section and disassemble it
        data = exe.get_memory_mapped_image()[begin:end]
        for i in md.disasm(data, begin):
            line = str(i).rstrip().split()

            for opcode in opcodes:
                if opcode in line:
                    opcodeList.append(opcode)
                    break

            last_address = int(i.address)
            last_size = i.size
        #sometimes you need to skip some bytes
        begin = max(int(last_address),begin)+last_size+1
        if begin >= end:
            break
    
    return opcodeList

# Extract pefile 12000 opcodes (LSTM+CNN Input Version)
def ExtractPefile12000Opcodes(opcodes,exe):
    #Opcode LIst
    opcodeList = []
    #get main code section
    main_code = get_main_code_section(exe.sections, exe.OPTIONAL_HEADER.BaseOfCode)
    #define architecutre of the machine 
    md = Cs(CS_ARCH_X86, CS_MODE_32)
    md.detail = True
    last_address = 0
    last_size = 0
    #Beginning of code section
    begin = main_code.PointerToRawData
    #the end of the first continuous bloc of code
    end = begin+main_code.SizeOfRawData
    while True:
        #parse code section and disassemble it
        data = exe.get_memory_mapped_image()[begin:end]
        for i in md.disasm(data, begin):
            # print(i)
            line = str(i).rstrip().split()

            for opcode in opcodes:
                if opcode in line:
                    opcodeList.append(opcode)
                    if len(opcodeList) > 12000:
                        return opcodeList
                    break

            last_address = int(i.address)
            last_size = i.size
        #sometimes you need to skip some bytes
        begin = max(int(last_address),begin)+last_size+1
        if begin >= end:
            break
    
    return opcodeList

# Convert opcodes sequence list to BPE token list
def Tokenizer(vocab, opcodeSequenceList):
    opcodeSequenceList = [] + opcodeSequenceList
    tempStr = ''
    for i in range(len(opcodeSequenceList)):
        tempStr = tempStr + ' ' + opcodeSequenceList[i]
    opcodeSequenceList = tempStr

    greedyVocab1 = [] + vocab
    greedyVocab1.sort(key=len)
    greedyVocab1.reverse()

    for i in range(len(greedyVocab1)):
        tempStr = ' ' + greedyVocab1[i][0]
        for j in range(1, len(greedyVocab1[i])):
            tempStr = tempStr + ' ' + greedyVocab1[i][j]
        greedyVocab1[i] = tempStr + ' '

    greedyVocab2 = [] + vocab
    greedyVocab2.sort(key=len)
    greedyVocab2.reverse()

    for i in range(len(greedyVocab2)):
        tempStr = '  ' + greedyVocab2[i][0]
        for j in range(1, len(greedyVocab2[i])):
            tempStr = tempStr + greedyVocab2[i][j]
        greedyVocab2[i] = tempStr + '  '
        
    for i in range(len(greedyVocab1)):
        opcodeSequenceList = opcodeSequenceList.replace(greedyVocab1[i], greedyVocab2[i])
    
    tempList = opcodeSequenceList.split(' ')
    tokenList = []
    for i in range(len(tempList)):
        if tempList[i] != '':                
            tokenList.append(tempList[i])
    
    return tokenList

# A function that matches the BPE Token list to Token Id
def TokenIdMapping(vocab, BPESequenceList):
    vocab2 = [] + vocab
    for i in range(len(vocab2)):
        tempStr = vocab2[i][0]
        for j in range(1, len(vocab2[i])):
            tempStr = tempStr + vocab2[i][j]
        vocab2[i] = tempStr

    vocab_number = [len(vocab) - i for i in range(len(vocab))]
    IdList = []
    
    for i in range(len(BPESequenceList)):
        IdList.append(vocab_number[vocab2.index(BPESequenceList[i])])
        
    return IdList
