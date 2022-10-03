# NB: <codegen> get all subdir in the generated folder.
MACRO(SUBDIRLIST result curdir)
  # code from https://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(SPUTNIK_GENERATED_SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/generated)

# NB: <codegen> add_subdirectory here for custom_spmm|sddmm
add_subdirectory(custom_spmm)
add_subdirectory(custom_sddmm)


# NB: <codegen> add custom_spmm|sddmm header install here.
install(FILES "custom_spmm/cuda_custom_spmm.h" DESTINATION "${INSTALL_BASE}/custom_spmm")
install(FILES "custom_sddmm/cuda_custom_sddmm.h" DESTINATION "${INSTALL_BASE}/custom_sddmm")
