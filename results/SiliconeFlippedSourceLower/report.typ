
#import "template.typ": *

#show: project.with(
  title: [Chroma Simulation \ SiliconeFlippedSourceLower \ silicon-lower-8reflector],
  authors: ((name: "Pocar Lab", email: "Lab 21", affiliation: "nEXO Collaboration"),),
  date: "2023-07-22",
)

= Experiment Details

- Experiment Name: SiliconeFlippedSourceLower
- Number of Particles: 1000000
- Random Seed: 1042
- Run ID: 1
- Excluded Geometry: []
- PTE: 0.0069 ± 0.0001.

#pagebreak()
= Material Properties

== Bulk Materials

#table(
  columns: 5,
  [name], [refractive_index], [absorption_length], [scattering_length], [density],
  [liquid xenon], [1.69], [350000000000.0], [35000000000000], [2.942],
  [copper], [0.9733], [0.05], [1000000], [8.96],
  [silicon], [0.682], [1000000.0], [10000000], [2.329],
  [steel], [1.29], [100000000.0], [1000000000], [7.75],
  [teflon], [0.0], [100000000.0], [1000000000], [7.75],
  [silica], [1.644], [100000000.0], [1000000000], [2.202],
  [aluminum], [0.09216], [1000000.0], [10000000], [2.7],
  [killing material], [1.0], [100000000.0], [1000000000], [0.0],
)

#table(
  columns: 6,
  [name], [abs(r_i_error)], [eta], [k], [abs(eta_error)], [abs(k_error)],
  [liquid xenon], [0.0], [0.0], [0.0], [0], [0],
  [copper], [0.0], [0.972], [1.5004], [0], [0],
  [silicon], [0.0], [0.83987], [1.9019], [0], [0],
  [steel], [0.0], [0.0], [0.0], [0], [0],
  [teflon], [0.0], [0.0], [0.0], [0], [0],
  [silica], [0.0], [0.0], [0.0], [0], [0],
  [aluminum], [0.0], [0.09216], [1.9217], [0], [0],
  [killing material], [0.0], [0.0], [0.0], [0], [0],
)

== Surface Properties

#table(
  columns: 6,
  [name], [outer_mat], [inner_mat], [model_id], [reflect_specular], [reflect_diffuse],
  [Al-Xe], [liquid xenon], [aluminum], [4], [0.0], [0.0],
  [Cu-Xe], [liquid xenon], [copper], [4], [0.0], [0.0],
  [FBK HD3], [liquid xenon], [silicon], [5], [0.0], [0.0],
  [killing surface], [None], [None], [8], [0.0], [0.0],
  [silicon-Xe], [liquid xenon], [silicon], [4], [0.7], [0.3],
  [Steel-Xe], [liquid xenon], [steel], [4], [0.0], [0.0],
  [teflon-Xe], [liquid xenon], [teflon], [9], [0.0], [0.0],
  [teflon-Xe], [liquid xenon], [teflon], [9], [0.0], [1.0],
)

= Results

== Photon Transmission Efficiency

The Photon Transmission Efficiency (PTE) for this experiment was 
0.0069 ± 0.0001.

== Tallies

#table(
  columns: (auto, auto),
  [Tally], [Count],
  [NO_HIT], [94600],
[BULK_ABSORB], [8215],
[SURFACE_DETECT], [6879],
[SURFACE_ABSORB], [889862],
[REFLECT_DIFFUSE], [0],
[REFLECT_SPECULAR], [444],

)
#pagebreak()
= Plots

#let plot_files = ("plots/detected_photon_tracks,_seed_1042,_run_1_seed_1042_run_1.png", "plots/position_distribution_seed_1042_run_1.png", "plots/incident_vs_emission_angle_seed_1042_run_1.png", "plots/reflection_angle_distribution_seed_1042_run_1.png", "plots/incident_angle_distribution_seed_1042_run_1.png", "plots/emission_angle_distribution_seed_1042_run_1.png")


#grid(
  columns: 2,
  gutter: 10pt,
  ..plot_files.map(file => {
    figure(
      image(file, width: 100%),
      caption: file.split("/").at(-1).replace("_", " ").replace(".png", "")
    )
  })
)


