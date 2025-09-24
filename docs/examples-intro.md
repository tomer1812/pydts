# Examples

In this section, we present the main functionalities of *PyDTS* through usage examples.  
We demonstrate how to:

1. Use `EventTimesSampler` to simulate discrete-time survival data with competing events, including random and hazard-based censoring.  
2. Perform estimation and prediction with `TwoStagesFitter` and `DataExpansionFitter`.  
3. Apply evaluation metrics.  
4. Add regularization.  
5. Handle small sample sizes using `TwoStagesFitterExact`.  
6. Conduct screening with `SISTwoStagesFitter`.  
7. Carry out model selection procedures.  
8. Propose data-preprocessing strategies to mitigate estimation errors that arise when the number of observed events is too small at specific times.
9. Work with a simulated hospitalization length-of-stay use case.  

Note - The figures included in the module `pydts.example_utils` are provided solely for visualization in the documentation and are not generalizable to arbitrary datasets.
